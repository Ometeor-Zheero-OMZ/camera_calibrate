use std::time::Instant;

use opencv::core::{
    TermCriteria,
    TermCriteria_Type,
    Size
};

use camera_calibration::{CameraCalibration, CameraCalibrationTrait};
use file::File;
mod camera_calibration;
mod file;

const FILE_FORMAT: &str = "jpeg";
const UNDISTORT_IMG_PATH: &str = "./img/calib02.jpeg";
const RESULT_IMG_PATH: &str = "./out/result.jpeg";
const JSON_PATH: &str = "./out/calibration.json";

fn main() -> opencv::Result<()> {
    File::create_out_dir();

    let start_time = Instant::now();

    let chessboard_size = Size::new(9, 6);
    let frame_size = Size::new(1440, 1080);
    
    let image_paths = File::get_image_paths("./img");
    let criteria = TermCriteria::new(
        (TermCriteria_Type::COUNT as i32) + (TermCriteria_Type::EPS as i32),
        30,
        0.001,
    )?;

    let (obj_points, img_points) =
        CameraCalibration::detect_chessboard_corners(&image_paths, chessboard_size, criteria)?;

    let (camera_matrix, dist_coeffs, rvecs, tvecs) =
        CameraCalibration::calibrate_camera(&obj_points, &img_points, frame_size, criteria)?;

    CameraCalibration::undistort_image(&camera_matrix, &dist_coeffs)?;

    let error = CameraCalibration::compute_reprojection_error(&obj_points, &img_points, &rvecs, &tvecs, &camera_matrix, &dist_coeffs)?;
    println!("Total Error: {}", error);

    if let Err(e) = CameraCalibration::save_to_json(&camera_matrix, &dist_coeffs, &rvecs, &tvecs, error, JSON_PATH) {
        eprintln!("Failed to save to json: {}", e);
    }

    let duration = start_time.elapsed();
    println!("Processing time: {:?}", duration);

    Ok(())
}