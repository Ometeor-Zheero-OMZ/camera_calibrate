use std::fs::File;
use std::io::Write;

use serde::{Serialize, Deserialize};
use serde_json;
use opencv::{
    core::{Mat, Vector},
    prelude::*
};

#[derive(Serialize, Deserialize)]
pub struct CameraCalibration {
    camera_matrix: Vec<Vec<f64>>,
    distortion_parameters: Vec<f64>,
    rotation_vectors: Vec<Vec<f64>>,
    translation_vectors: Vec<Vec<f64>>,
    total_error: f64,
}

pub trait CameraCalibrationTrait {
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
