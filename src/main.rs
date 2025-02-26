use std::time::Instant;

use opencv::core::{
    TermCriteria,
    TermCriteria_Type,
    Size
};
use clap::Parser;

use camera_calibration::{CameraCalibration, CameraCalibrationTrait};
use file::CustomFile;
use command_line::{Args, CalibrationPattern};
mod camera_calibration;
mod file;
mod command_line;

// IMAGE FORMAT READ & DETECTED 
const FILE_FORMAT: &str = "jpeg";

// DISPLAY IMAGE WINDOW SIZE
const WINDOW_TITLE: &str = "Chessboard Corners Detection";
const GUI_WINDOW_WIDTH: i32 = 900;
const GUI_WINDOW_HEIGHT: i32 = 700;
const WAIT_KEY_DELAY: i32 = 1000;

// TEXT
const TEXT_POINT: (i32, i32) = (10, 100);
const TEXT_FONT_SCALE: f64 = 3.0;
const TEXT_COLOR: (f64, f64, f64, f64) = (0.0, 255.0, 0.0, 0.0); // GREEN

fn main() -> opencv::Result<()> {
    CustomFile::create_out_dir(None);

    let args = Args::parse();
    let _ = match args.calibrate {
        Some(CalibrationPattern::ChessBoard) => chessboard(),
        Some(CalibrationPattern::SymmetricCircleGrid) => symmetric_circle_grid(),
        Some(CalibrationPattern::AsymmetricCircleGrid) => unimplemented!(),
        Some(CalibrationPattern::ChArUco) => unimplemented!(),
        None => unimplemented!(),
    };

    Ok(())
}

fn chessboard() -> opencv::Result<()> {
    // FILE & DIRECTORY PATH
    const CHESSBOARD_DIRECTORY_NAME: &str = "chessboard";
    const FAILED_READ_IMAGES_PATH: &str = "./out/chessboard/failed_read_files.json";
    const READ_DATASET_PATH: &str = "./img/chessboard_dataset";
    const UNDISTORT_IMAGE_PATH: &str = "./img/chessboard_dataset/calib04.jpeg";
    const RESULT_IMAGE_PATH: &str = "./out/chessboard/result.jpeg";
    const CALIBRATION_JSON_PATH: &str = "./out/chessboard/calibration.json";
    // CAMERA CALIBRATION PARAMETERS
    const CHESSBOARD_SIZE: (i32, i32) = (9, 6);
    const FRAME_WIDTH: i32 = 1440;
    const FRAME_HEIGHT: i32 = 1080;
    const CRITERIA_MAX_COUNT: i32 = 30;
    const CRITERIA_EPS: f64 = 0.001;
    // CAMERA CALIBRATION PARAMETERS
    const CORNER_SUB_PIX_WINDOW_WIDTH: i32 = 11;
    const CORNER_SUB_PIX_WINDOW_HEIGHT: i32 = 11;
    const CORNER_SUB_PIX_ZERO_ZONE: i32 = -1;

    let start_time = Instant::now();

    let chessboard_size = Size::new(CHESSBOARD_SIZE.0, CHESSBOARD_SIZE.1);
    let image_paths = CustomFile::get_image_paths(READ_DATASET_PATH);
    let criteria = TermCriteria::new(
        (TermCriteria_Type::COUNT as i32) + (TermCriteria_Type::EPS as i32),
        CRITERIA_MAX_COUNT,
        CRITERIA_EPS,
    )?;

    let mut read_image_cnt = 0;
    let (obj_points, img_points) =
        CameraCalibration::detect_chessboard_corners(
            &image_paths,
            chessboard_size,
            criteria,
            &mut read_image_cnt,
            FAILED_READ_IMAGES_PATH,
            CORNER_SUB_PIX_WINDOW_WIDTH,
            CORNER_SUB_PIX_WINDOW_HEIGHT,
            CORNER_SUB_PIX_ZERO_ZONE,
        )?;

    let frame_size = Size::new(FRAME_WIDTH, FRAME_HEIGHT);
    let (camera_matrix, dist_coeffs, rvecs, tvecs) =
        CameraCalibration::calibrate_camera(&obj_points, &img_points, frame_size, criteria)?;

    CameraCalibration::undistort_image(&camera_matrix, &dist_coeffs, UNDISTORT_IMAGE_PATH, RESULT_IMAGE_PATH, CHESSBOARD_DIRECTORY_NAME)?;

    let error = CameraCalibration::compute_reprojection_error(&obj_points, &img_points, &rvecs, &tvecs, &camera_matrix, &dist_coeffs)?;
    println!("Total Error: {}", error);

    if let Err(e) = CameraCalibration::save_to_json(&camera_matrix, &dist_coeffs, &rvecs, &tvecs, error, CALIBRATION_JSON_PATH) {
        eprintln!("Failed to save to json: {}", e);
    }

    let duration = start_time.elapsed();
    println!("Processing time: {:?}", duration);

    Ok(())
}

fn symmetric_circle_grid() -> opencv::Result<()> {
    // FILE & DIRECTORY PATH
    const CIRCLE_GRID_DIRECTORY_NAME: &str = "circle_grid";
    const FAILED_READ_IMAGES_PATH: &str = "./out/circle_grid/failed_read_files.json";
    const READ_DATASET_PATH: &str = "./img/circle_grid_dataset";
    const UNDISTORT_IMAGE_PATH: &str = "./img/circle_grid_dataset/calib04.jpeg";
    const RESULT_IMAGE_PATH: &str = "./out/circle_grid/result.jpeg";
    const CALIBRATION_JSON_PATH: &str = "./out/circle_grid/calibration.json";
    // CAMERA CALIBRATION PARAMETERS
    const CHESSBOARD_SIZE: (i32, i32) = (9, 6);
    const FRAME_WIDTH: i32 = 1440;
    const FRAME_HEIGHT: i32 = 1080;
    const CRITERIA_MAX_COUNT: i32 = 30;
    const CRITERIA_EPS: f64 = 0.001;

    let start_time = Instant::now();

    let chessboard_size = Size::new(CHESSBOARD_SIZE.0, CHESSBOARD_SIZE.1);
    let image_paths = CustomFile::get_image_paths(READ_DATASET_PATH);
    let criteria = TermCriteria::new(
        (TermCriteria_Type::COUNT as i32) + (TermCriteria_Type::EPS as i32),
        CRITERIA_MAX_COUNT,
        CRITERIA_EPS,
    )?;

    let mut read_image_cnt = 0;
    let (obj_points, img_points) =
        CameraCalibration::detect_circle_grid(&image_paths, chessboard_size, &mut read_image_cnt, FAILED_READ_IMAGES_PATH)?;

    let frame_size = Size::new(FRAME_WIDTH, FRAME_HEIGHT);
    println!("Reached here: Before calibrate_camera");
    let (camera_matrix, dist_coeffs, rvecs, tvecs) =
        CameraCalibration::calibrate_camera(&obj_points, &img_points, frame_size, criteria)?;

    println!("Reached here: Before undistort_image");
    CameraCalibration::undistort_image(&camera_matrix, &dist_coeffs, UNDISTORT_IMAGE_PATH, RESULT_IMAGE_PATH, CIRCLE_GRID_DIRECTORY_NAME)?;

    let error = CameraCalibration::compute_reprojection_error(&obj_points, &img_points, &rvecs, &tvecs, &camera_matrix, &dist_coeffs)?;
    println!("Total Error: {}", error);

    if let Err(e) = CameraCalibration::save_to_json(&camera_matrix, &dist_coeffs, &rvecs, &tvecs, error, CALIBRATION_JSON_PATH) {
        eprintln!("Failed to save to json: {}", e);
    }

    let duration = start_time.elapsed();
    println!("Processing time: {:?}", duration);

    Ok(())
}