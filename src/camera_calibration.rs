use std::fs::File;
use std::io::Write;

use serde::{Serialize, Deserialize};
use serde_json;
use opencv::{
    calib3d,
    core::{self, Mat, Point2f, Point3f, Size, Vector},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};
use rayon::prelude::*;

use crate::{RESULT_IMG_PATH, UNDISTORT_IMG_PATH};

#[derive(Serialize, Deserialize)]
pub struct CameraCalibration {
    camera_matrix: Vec<Vec<f64>>,
    distortion_parameters: Vec<f64>,
    rotation_vectors: Vec<Vec<f64>>,
    translation_vectors: Vec<Vec<f64>>,
    total_error: f64,
}

pub trait CameraCalibrationTrait {
    /// チェスボードのコーナー検出 & 精緻化
    fn detect_chessboard_corners(
        image_paths: &[std::path::PathBuf],
        chessboard_size: Size,
        criteria: core::TermCriteria,
    ) -> opencv::Result<(Vector<Vector<Point3f>>, Vector<Vector<Point2f>>)>;

    /// カメラキャリブレーション
    fn calibrate_camera(
        obj_points: &Vector<Vector<Point3f>>,
        img_points: &Vector<Vector<Point2f>>,
        frame_size: Size,
        criteria: core::TermCriteria,
    ) -> opencv::Result<(Mat, Mat, Vector<Mat>, Vector<Mat>)>;

    /// 画像の歪み補正
    fn undistort_image(camera_matrix: &Mat, dist_coeffs: &Mat) -> opencv::Result<()>;

    /// 再投影誤差を計算
    fn compute_reprojection_error(
        obj_points: &Vector<Vector<Point3f>>,
        img_points: &Vector<Vector<Point2f>>,
        rvecs: &Vector<Mat>,
        tvecs: &Vector<Mat>,
        camera_matrix: &Mat,
        dist_coeffs: &Mat,
    ) -> opencv::Result<f64>;

    /// カメラキャリブレーション結果をJSON形式で保存
    fn save_to_json(
        camera_matrix: &Mat,
        dist_coeffs: &Mat,
        rvecs: &Vector<Mat>,
        tvecs: &Vector<Mat>,
        error: f64,
        filename: &str
    ) -> std::io::Result<()>;
}

impl CameraCalibrationTrait for CameraCalibration {
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
    
                // Read each file and display the filename on the window
                if let Some(filename) = image_path.file_name().and_then(|f| f.to_str()) {
                    let text = format!("{}", filename);
                    let org = core::Point::new(10, 100);
                    let font_face = imgproc::FONT_HERSHEY_SIMPLEX;
                    let font_scale = 3.0;
                    let color = core::Scalar::new(0.0, 255.0, 0.0, 0.0); // GREEN TEXT
                    let thickness = 2;
                    imgproc::put_text(&mut img_clone, &text, org, font_face, font_scale, color, thickness, imgproc::LINE_AA, false)?;
                }
    
                highgui::imshow("img", &img_clone)?;
                highgui::wait_key(1000)?;
            }
        }
    
        highgui::destroy_all_windows()?;
        Ok((obj_points, img_points))
    }

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

    fn compute_reprojection_error(
        obj_points: &Vector<Vector<Point3f>>,
        img_points: &Vector<Vector<Point2f>>,
        rvecs: &Vector<Mat>,
        tvecs: &Vector<Mat>,
        camera_matrix: &Mat,
        dist_coeffs: &Mat,
    ) -> opencv::Result<f64> {
        let errors: Vec<f64> = (0..obj_points.len())
            .into_par_iter()
            .map(|i| {
                let mut img_points2 = Vector::<Point2f>::new();
                let mut jacobian = Mat::default();

                if calib3d::project_points(
                    &obj_points.get(i).unwrap(),
                    &rvecs.get(i).unwrap(),
                    &tvecs.get(i).unwrap(),
                    camera_matrix,
                    dist_coeffs,
                    &mut img_points2,
                    &mut jacobian,
                    0.0,
                ).is_err() {
                    return 0.0;
                }

                let img_points_vec = img_points.get(i).unwrap().to_vec();
                if let Ok(norm) = core::norm(
                    &Mat::from_slice(&img_points_vec).unwrap().reshape(1, img_points_vec.len() as i32).unwrap(),
                    core::NORM_L2,
                    &Mat::default(),
                ) {
                    norm
                } else {
                    0.0
                }
            })
            .collect();

        let mean_error = errors.iter().sum::<f64>() / obj_points.len() as f64;
        Ok(mean_error)
    }

    fn save_to_json(
        camera_matrix: &Mat,
        dist_coeffs: &Mat,
        rvecs: &Vector<Mat>,
        tvecs: &Vector<Mat>,
        error: f64,
        filename: &str,
    ) -> std::io::Result<()> {
        //  カメラ行列をVec<Vec<f64>>に変換
        let rows = camera_matrix.rows() as usize;
        let cols = camera_matrix.cols() as usize;
        let mut camera_matrix_vec = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                camera_matrix_vec[i][j] = *camera_matrix.at_2d::<f64>(i as i32, j as i32).unwrap();
            }
        }

        let dist_coeffs_vec: Vec<f64> = (0..dist_coeffs.total())
            .map(|i| *dist_coeffs.at::<f64>(i as i32).unwrap())
            .collect();

        // 回転ベクトルと並進ベクトルをVec<Vec<f64>>に変換
        let mut rotation_vectors = Vec::new();
        let mut translation_vectors = Vec::new();
        for rvec in rvecs {
            let rvec_vec = (0..rvec.total() as usize)
                .map(|i| *rvec.at::<f64>(i as i32).unwrap())
                .collect::<Vec<f64>>();
            rotation_vectors.push(rvec_vec);
        }
        for tvec in tvecs {
            let tvec_vec = (0..tvec.total() as usize)
                .map(|i| *tvec.at::<f64>(i as i32).unwrap())
                .collect::<Vec<f64>>();
            translation_vectors.push(tvec_vec);
        }

        let calibration = CameraCalibration {
            camera_matrix: camera_matrix_vec,
            distortion_parameters: dist_coeffs_vec,
            rotation_vectors,
            translation_vectors,
            total_error: error,
        };

        let json_string = serde_json::to_string_pretty(&calibration)?;
        let mut file = File::create(filename)?;
        file.write_all(json_string.as_bytes())?;

        Ok(())
    }
}
