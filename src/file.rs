use std::fs;
use std::path::PathBuf;

use crate::FILE_FORMAT;

pub struct File {}

impl File {
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
}