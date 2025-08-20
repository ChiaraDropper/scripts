#include "ArducamTOFCamera.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include <filesystem>
#include <unordered_map>
#include <set>

namespace fs = std::filesystem;

#define MAX_DISTANCE 4000
#define CONFIDENCE_VALUE 60

using namespace Arducam;

int in_count = 0, out_count = 0, tot_persone = 0, next_track_id = 0;
double soglia = 0.0;
int frame_id = 0;
int nframe = 0;
int visualize_frame_counter = 0;

struct TrackedObject {
    int id;
    Eigen::Vector3d center;
    int age;
    int missing_frames;
};

std::unordered_map<int, TrackedObject> active_tracks;
std::unordered_map<int, double> previous_x_position;

bool getControl(ArducamTOFCamera& tof, Control mode, float& val, float alpha = 1.0) {
    int tmp = 0;
    if (tof.getControl(mode, &tmp) != 0) return false;
    val = tmp / alpha;
    return true;
}

bool initCamera(ArducamTOFCamera& tof, const char* cfg_path) {
    if (cfg_path && tof.openWithFile(cfg_path)) return false;
    if (!cfg_path && tof.open(Connection::CSI)) return false;
    if (tof.start(FrameType::DEPTH_FRAME)) return false;
    tof.setControl(Control::RANGE, MAX_DISTANCE);
    return true;
}

ArducamFrameBuffer* acquireFrame(ArducamTOFCamera& tof) {
    return tof.requestFrame(500);
}


void initializeFrameID(const std::string& path = "dataset_new_try/training/velodyne") {
    if (fs::exists(path)) {
        frame_id = std::distance(fs::directory_iterator(path), fs::directory_iterator{});
    }
}


std::shared_ptr<open3d::geometry::PointCloud> generatePointCloud(ArducamTOFCamera& tof, ArducamFrameBuffer* frame, Eigen::Matrix4d transform) {
    Arducam::FrameFormat format;
    frame->getFormat(FrameType::DEPTH_FRAME, format);

    float* depth = (float*)frame->getData(FrameType::DEPTH_FRAME);
    float* conf = (float*)frame->getData(FrameType::CONFIDENCE_FRAME);
    std::vector<float> filtered(format.width * format.height);

    for (int i = 0; i < format.width * format.height; ++i)
        filtered[i] = (conf[i] >= CONFIDENCE_VALUE) ? depth[i] : 0;

    open3d::geometry::Image depth_img;
    depth_img.Prepare(format.width, format.height, 1, 4);
    memcpy(depth_img.data_.data(), filtered.data(), filtered.size() * sizeof(float));

    float fx, fy, cx, cy;
    getControl(tof, Control::INTRINSIC_FX, fx, 100);
    getControl(tof, Control::INTRINSIC_FY, fy, 100);
    getControl(tof, Control::INTRINSIC_CX, cx, 100);
    getControl(tof, Control::INTRINSIC_CY, cy, 100);

    auto cloud = open3d::geometry::PointCloud::CreateFromDepthImage(
        depth_img, {format.width, format.height, fx, fy, cx, cy}, Eigen::Matrix4d::Identity(), 1000.0, 2.5);
    cloud->Transform(transform);

    std::vector<size_t> keep;
    for (size_t i = 0; i < cloud->points_.size(); ++i)
        if (cloud->points_[i](2) > -2000)
            keep.push_back(i);

    return cloud->SelectByIndex(keep);
}

std::vector<int> clusterAndColor(std::shared_ptr<open3d::geometry::PointCloud>& cloud, std::vector<Eigen::Vector3d>& colors) {
    std::vector<int> labels = cloud->ClusterDBSCAN(230, 100, true);
    colors.resize(cloud->points_.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        int l = labels[i];
        if (l < 0) colors[i] = Eigen::Vector3d(0.5, 0.5, 0.5);
        else colors[i] = Eigen::Vector3d((l%5)/5.0, ((l+2)%5)/5.0, ((l+4)%5)/5.0);
    }
    return labels;
}

void performTracking(std::shared_ptr<open3d::geometry::PointCloud>& cloud, const std::vector<int>& labels) {
    std::unordered_map<int, Eigen::Vector3d> centroids;
    std::unordered_map<int, int> sizes;

    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == -1) continue;
        centroids[labels[i]] += cloud->points_[i];
        sizes[labels[i]]++;
    }
    for (auto& [id, sum] : centroids) sum /= sizes[id];

    std::unordered_map<int, int> match;
    std::set<int> used;
    double max_dist = 1000.0;

    for (const auto& [label, center] : centroids) {
        double best_dist = max_dist; int best_id = -1;
        for (const auto& [id, track] : active_tracks) {
            double d = (center - track.center).norm();
            if (d < best_dist && used.find(id) == used.end()) {
                best_dist = d; best_id = id;
            }
        }
        if (best_id != -1) {
            match[label] = best_id;
            used.insert(best_id);
            auto& t = active_tracks[best_id];
            t.center = center; t.age++; t.missing_frames = 0;
        } else {
            int nid = next_track_id++;
            match[label] = nid;
            active_tracks[nid] = {nid, center, 1, 0};
        }
    }

    for (const auto& [label, id] : match) {
        const auto& center = active_tracks[id].center;
        double curr_x = center(0);
        double prev_x = previous_x_position.count(id) ? previous_x_position[id] : curr_x;
        previous_x_position[id] = curr_x;

        if (prev_x < soglia && curr_x >= soglia) {
            out_count++; tot_persone--;
            std::cout << "[OUT] ID " << id << " | Tot OUT: " << out_count << " | Tot IN: " << tot_persone << "\n";
        }
        if (prev_x > soglia && curr_x <= soglia) {
            in_count++; tot_persone++;
            std::cout << "[IN]  ID " << id << " | Tot IN:  " << in_count << " | Tot IN: " << tot_persone << "\n";
        }
    }
}

void saveToKittiFormat(const std::shared_ptr<open3d::geometry::PointCloud>& cloud, const std::string& class_label) {
    fs::create_directories("dataset_new_try/training/velodyne");
    fs::create_directories("dataset_new_try/training/label_2");

    std::ostringstream base;
    base << std::setw(6) << std::setfill('0') << frame_id++;

    std::ofstream bin("dataset_new_try/training/velodyne/" + base.str() + ".bin", std::ios::binary);
    for (const auto& pt : cloud->points_) {
        float x = pt(0), y = pt(1), z = pt(2), i = 0.0f;
        bin.write(reinterpret_cast<char*>(&x), sizeof(float));
        bin.write(reinterpret_cast<char*>(&y), sizeof(float));
        bin.write(reinterpret_cast<char*>(&z), sizeof(float));
        bin.write(reinterpret_cast<char*>(&i), sizeof(float));
    }
    bin.close();

    std::ofstream label("dataset_new_try/training/label_2/" + base.str() + ".txt");
    label << class_label << "\n";
    std::cout << "[INFO] Salvato frame " << base.str() << " con etichetta \"" << class_label << "\"\n";
    label.close();
    nframe++;
}

bool pc_loop(ArducamTOFCamera& tof, std::shared_ptr<open3d::geometry::PointCloud>& pcd, Eigen::Matrix4d transform) {
    auto* frame = acquireFrame(tof);
    visualize_frame_counter++;
    if (!frame) return true;
    auto cloud = generatePointCloud(tof, frame, transform);
    if (!cloud || cloud->points_.empty()) { tof.releaseFrame(frame); return true; }

    std::vector<Eigen::Vector3d> colors;
    //auto labels = clusterAndColor(cloud, colors);

    //performTracking(cloud, labels);

    cloud->colors_ = colors;
    pcd->points_ = cloud->points_;
    pcd->colors_ = cloud->colors_;
    
    if (visualize_frame_counter % 10 == 0) saveToKittiFormat(cloud, "two_people");

    tof.releaseFrame(frame);
    return true;
}

int main(int argc, char* argv[]) {
    open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Error);

    ArducamTOFCamera tof;
    const char* cfg_path = (argc > 1) ? argv[1] : nullptr;
    if (!initCamera(tof, cfg_path)) return -1;

    initializeFrameID();
    std::cout << "[INFO] Starting from frame_id = " << frame_id << std::endl;


    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m << 1, 0, 0, 0,
         0, -1, 0, 0,
         0, 0, -1, 0,
         0, 0, 0, 1;

    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Live Viewer", 1280, 720);
    vis.AddGeometry(open3d::geometry::TriangleMesh::CreateCoordinateFrame(500.0));
    vis.AddGeometry(pcd);

    while (pc_loop(tof, pcd, m)) {
        vis.UpdateGeometry(pcd);
        vis.PollEvents();
        vis.UpdateRender();
        if (nframe == 50){
            return 0;
        }
    }

    vis.DestroyVisualizerWindow();
    tof.stop();
    tof.close();
    return 0;
}


//FRAME DA 0 A 549 -> NONE

