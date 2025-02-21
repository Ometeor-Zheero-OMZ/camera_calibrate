use std::fs;
use std::path::Path;

use crate::FILE_FORMAT;

pub struct File {}

impl File {
    /// 出力ディレクトリを作成する
    pub fn create_out_dir() {
        let dir_name = "out";
        if !Path::new(dir_name).exists() {
            fs::create_dir(dir_name).expect("Failed to create directory");
            println!("Directory '{}' created", dir_name);
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