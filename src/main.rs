use std::fs;
use std::path::Path;
use std::time::Instant;

use opencv::{
    calib3d,
    core::{self, Mat, Point2f, Point3f, Size, Vector},
    highgui, imgcodecs, imgproc, prelude::*,
};

use json::{CameraCalibration, CameraCalibrationTrait};
mod json;

const FILE_FORMAT: &str = "jpeg";
const UNDISTORT_IMG_PATH: &str = "./img/calib4.jpeg";
const RESULT_IMG_PATH: &str = "./out/result.jpeg";
const JSON_PATH: &str = "./out/calibration.json";

fn main() -> opencv::Result<()> {
    create_out_dir();

    let start_time = Instant::now();

    let chessboard_size = Size::new(9, 6);
    let frame_size = Size::new(1440, 1080);
    
    let image_paths = get_image_paths("./img");
    let criteria = core::TermCriteria::new(
        (core::TermCriteria_Type::COUNT as i32) + (core::TermCriteria_Type::EPS as i32),
        30,
        0.001,
    )?;

    let (obj_points, img_points) = detect_chessboard_corners(&image_paths, chessboard_size, criteria)?;

    let (camera_matrix, dist_coeffs, rvecs, tvecs) =
        calibrate_camera(&obj_points, &img_points, frame_size, criteria)?;

    undistort_image(&camera_matrix, &dist_coeffs)?;

    let error = compute_reprojection_error(&obj_points, &img_points, &rvecs, &tvecs, &camera_matrix, &dist_coeffs)?;
    println!("Total Error: {}", error);

    if let Err(e) = CameraCalibration::save_to_json(&camera_matrix, &dist_coeffs, &rvecs, &tvecs, error, JSON_PATH) {
        eprintln!("Failed to save to json: {}", e);
    }

    let duration = start_time.elapsed();
    println!("Processing time: {:?}", duration);

    Ok(())
}

/// 出力ディレクトリを作成する
fn create_out_dir() {
    let dir_name = "out";
    if !Path::new(dir_name).exists() {
        fs::create_dir(dir_name).expect("Failed to create directory");
        println!("Directory '{}' created", dir_name);
    }
}

/// 指定したディレクトリ内の画像パスを取得する
fn get_image_paths(dir: &str) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().map(|ext| ext == FILE_FORMAT).unwrap_or(false))
        .map(|entry| entry.path())
        .collect()
}

/// チェスボードのコーナー検出 & 精緻化
fn detect_chessboard_corners(
    image_paths: &[std::path::PathBuf],
    chessboard_size: Size,
    criteria: core::TermCriteria,
) -> opencv::Result<(Vector<Vector<Point3f>>, Vector<Vector<Point2f>>)> {
    let mut objp = Vector::<Point3f>::new();
    for i in 0..chessboard_size.height {
        for j in 0..chessboard_size.width {
            objp.push(Point3f::new(j as f32, i as f32, 0.0));
        }
    }

    let mut obj_points = Vector::<Vector<Point3f>>::new();
    let mut img_points = Vector::<Vector<Point2f>>::new();

    highgui::named_window("img", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("img", 800, 600)?;

    for image_path in image_paths {
        let img = imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        let mut gray = Mat::default();
        imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut corners = Vector::<Point2f>::new();
        let found = calib3d::find_chessboard_corners(
            &gray,
            chessboard_size,
            &mut corners,
            calib3d::CALIB_CB_ADAPTIVE_THRESH
                | calib3d::CALIB_CB_FAST_CHECK
                | calib3d::CALIB_CB_NORMALIZE_IMAGE,
        )?;

        if found {
            obj_points.push(objp.clone());

            let mut refined_corners = corners.clone();
            imgproc::corner_sub_pix(
                &gray,
                &mut refined_corners,
                Size::new(11, 11),
                Size::new(-1, -1),
                criteria,
            )?;
            img_points.push(refined_corners);

            let mut img_clone = img.clone();
            calib3d::draw_chessboard_corners(&mut img_clone, chessboard_size, &corners, found)?;
            highgui::imshow("img", &img_clone)?;
            highgui::wait_key(1000)?;
        }
    }

    highgui::destroy_all_windows()?;
    Ok((obj_points, img_points))
}

/// カメラキャリブレーション
fn calibrate_camera(
    obj_points: &Vector<Vector<Point3f>>,
    img_points: &Vector<Vector<Point2f>>,
    frame_size: Size,
    criteria: core::TermCriteria,
) -> opencv::Result<(Mat, Mat, Vector<Mat>, Vector<Mat>)> {
    let mut camera_matrix = Mat::default();
    let mut dist_coeffs = Mat::default();
    let mut rvecs = Vector::<Mat>::new();
    let mut tvecs = Vector::<Mat>::new();

    let ret = calib3d::calibrate_camera(
        obj_points,
        img_points,
        frame_size,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        0,
        criteria,
    )?;

    // I leave this output because it may be useful for future cases where
    // the program needs to handle numerical values with extremely high precision such as microscopes.
    println!("Camera Calibrated: {}", ret);
    println!("Camera Matrix:\n{:?}", camera_matrix);
    println!("Distortion Parameters:\n{:?}", dist_coeffs);

    Ok((camera_matrix, dist_coeffs, rvecs, tvecs))
}

/// 画像の歪み補正
fn undistort_image(camera_matrix: &Mat, dist_coeffs: &Mat) -> opencv::Result<()> {
    let img = imgcodecs::imread(UNDISTORT_IMG_PATH, imgcodecs::IMREAD_COLOR)?;
    let size = img.size()?;
    let new_camera_matrix = Mat::default();
    let mut roi = core::Rect::default();

    calib3d::get_optimal_new_camera_matrix(
        camera_matrix,
        dist_coeffs,
        size,
        1.0,
        size,
        Some(&mut roi),
        false,
    )?;

    let mut dst = Mat::default();
    calib3d::undistort(&img, &mut dst, camera_matrix, dist_coeffs, &new_camera_matrix)?;

    imgcodecs::imwrite(RESULT_IMG_PATH, &dst, &Vector::new())?;
    Ok(())
}

/// 再投影誤差を計算
fn compute_reprojection_error(
    obj_points: &Vector<Vector<Point3f>>,
    img_points: &Vector<Vector<Point2f>>,
    rvecs: &Vector<Mat>,
    tvecs: &Vector<Mat>,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
) -> opencv::Result<f64> {
    let mut mean_error = 0.0;
    for i in 0..obj_points.len() {
        let mut img_points2 = Vector::<Point2f>::new();
        let mut jacobian = Mat::default();
        calib3d::project_points(
            &obj_points.get(i)?,
            &rvecs.get(i)?,
            &tvecs.get(i)?,
            camera_matrix,
            dist_coeffs,
            &mut img_points2,
            &mut jacobian,
            0.0,
        )?;

        let img_points_vec = img_points.get(i)?.to_vec();
        let error = core::norm(&Mat::from_slice(&img_points_vec)?
            .reshape(1, img_points_vec.len() as i32)?, core::NORM_L2, &Mat::default())?;
        mean_error += error;
    }

    Ok(mean_error / obj_points.len() as f64)
}
