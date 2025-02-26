use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use opencv::{core, Error as OpenCvError};
use serde_json::json;

use crate::FILE_FORMAT;

pub struct CustomFile {}

impl CustomFile {
    /// 出力ディレクトリを作成する
    pub fn create_out_dir(directory_path: Option<&str>) {
        let out_dir = PathBuf::from("out");

        if !out_dir.exists() {
            match fs::create_dir(&out_dir) {
                Ok(_) => println!("Directory '{}' created", out_dir.display()),
                Err(e) => {
                    eprintln!("Failed to create base directory '{}': {}", out_dir.display(), e);
                    return;
                }
            }
        }

        if let Some(path) = directory_path {
            let sub_dir = out_dir.join(path);
            if !sub_dir.exists() {
                match fs::create_dir(&sub_dir) {
                    Ok(_) => println!("Subdirectory '{}' created", sub_dir.display()),
                    Err(e) => eprintln!("Failed to create subdirectory '{}': {}", sub_dir.display(), e),
                }
            }
        }
    }

    /// 指定したディレクトリ内の画像パスを取得する
    pub fn get_image_paths(dir: &str) -> Vec<std::path::PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().map(|ext| ext == FILE_FORMAT).unwrap_or(false))
            .map(|entry| entry.path())
            .collect()
    }

    /// 画像の読み込みに失敗したファイルのリストをJSON形式で出力
    pub fn create_output_json(file_path: &str, json_data: Vec<String>) -> opencv::Result<(), OpenCvError> {
        // 失敗した画像のリストをJSON形式で出力
        if json_data.is_empty() {
            println!("No failed images to report.");
            return Ok(());
        }

        println!("Failed to read image: {:?}", json_data);

        // JSONデータを作成
        let failed_json = json!({ "failed_read_json": json_data });
        let pretty_json = match serde_json::to_string_pretty(&failed_json) {
            Ok(json) => json,
            Err(e) => {
                eprintln!("JSON serialize error: {}", e);
                return Err(OpenCvError::new(core::StsError, format!("JSON serialize error: {}", e)));
            }
        };

        // JSONファイルの書き込み
        println!("Writing JSON to: {}", file_path);

        // 出力先のディレクトリ作成
        let binding = PathBuf::from(file_path);
        let dir = binding.parent();
        if let Some(dir_path) = dir {
            if !dir_path.exists() {
                if let Err(e) = fs::create_dir_all(dir_path) {
                    eprintln!("Failed to create directory: {}", e);
                    return Err(OpenCvError::new(core::StsError, format!("Failed to create directory: {}", e)));
                }
            }
        }

        // ファイル作成と書き込み
        match File::create(file_path) {
            Ok(mut file) => {
                if let Err(e) = file.write_all(pretty_json.as_bytes()) {
                    eprintln!("File write error: {}", e);
                    return Err(OpenCvError::new(core::StsError, format!("File write error: {}", e)));
                }
                println!("JSON successfully written.");
            }
            Err(e) => {
                eprintln!("File create error: {}", e);
                return Err(OpenCvError::new(core::StsError, format!("File create error: {}", e)));
            }
        }

        Ok(())
    }
}