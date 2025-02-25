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

// FILE
const FILE_FORMAT: &str = "jpeg";
const UNDISTORT_IMG_PATH: &str = "./img/circle_grid_dataset/calib04.jpeg";
const RESULT_IMG_PATH: &str = "./out/circle_grid/result.jpeg";
const JSON_PATH: &str = "./out/circle_grid/calibration.json";

// CAMERA CALIBRATION PARAMETERS
const CHESSBOARD_SIZE: (i32, i32) = (9, 6);
const FRAME_WIDTH: i32 = 1440;
const FRAME_HEIGHT: i32 = 1080;
const CRITERIA_MAX_COUNT: i32 = 30;
const CRITERIA_EPS: f64 = 0.001;
const CORNER_SUB_PIX_WINDOW_WIDTH: i32 = 11;
const CORNER_SUB_PIX_WINDOW_HEIGHT: i32 = 11;
const CORNER_SUB_PIX_ZERO_ZONE: i32 = -1;

// DISPLAY IMAGE WINDOW SIZE
const  WINDOW_TITLE: &str = "Chessboard Corners Detection";
const GUI_WINDOW_WIDTH: i32 = 900;
const GUI_WINDOW_HEIGHT: i32 = 700;
const WAIT_KEY_DELAY: i32 = 1000;

// TEXT
const TEXT_POINT: (i32, i32) = (10, 100);
const TEXT_FONT_SCALE: f64 = 3.0;
const TEXT_COLOR: (f64, f64, f64, f64) = (0.0, 255.0, 0.0, 0.0); // GREEN

fn main() -> opencv::Result<()> {
    File::create_out_dir(None);
    let start_time = Instant::now();

    let chessboard_size = Size::new(CHESSBOARD_SIZE.0, CHESSBOARD_SIZE.1);
    let image_paths = File::get_image_paths("./img/circle_grid_dataset");
    let criteria = TermCriteria::new(
        (TermCriteria_Type::COUNT as i32) + (TermCriteria_Type::EPS as i32),
        CRITERIA_MAX_COUNT,
        CRITERIA_EPS,
    )?;

    // let (obj_points, img_points) =
    //     CameraCalibration::detect_chessboard_corners(&image_paths, chessboard_size, criteria)?;

    let mut read_image_cnt = 0;
    let (obj_points, img_points) =
        CameraCalibration::detect_circle_grid(&image_paths, chessboard_size, &mut read_image_cnt)?;

    let frame_size = Size::new(FRAME_WIDTH, FRAME_HEIGHT);
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