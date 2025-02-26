use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(version, about, flatten_help = true)]
pub struct Args {
    /// select chessboard pattern
    #[arg(
        short = 'c',
        long = "calibrate",
        value_enum,
        default_value = "default",
        value_parser = validate_calibrate
    )]
    pub calibrate: Option<CalibrationPattern>,
}

fn validate_calibrate(val: &str) -> Result<CalibrationPattern, String> {
    match val.to_lowercase().as_str() {
        "chessboard" => Ok(CalibrationPattern::ChessBoard),
        "symmetric" => Ok(CalibrationPattern::SymmetricCircleGrid),
        "asymmetric" => Ok(CalibrationPattern::AsymmetricCircleGrid),
        "charuco" => Ok(CalibrationPattern::ChArUco),
        _ => Err(format!(
            "Invalid calibration pattern: '{}'. Allowed values are: chessboard, symmetric, asymmetric, charuco.",
            val
        ))
    }
}

#[derive(ValueEnum, Debug, Clone, PartialEq)]
pub enum CalibrationPattern {
    ChessBoard,
    SymmetricCircleGrid,
    AsymmetricCircleGrid,
    ChArUco,
}