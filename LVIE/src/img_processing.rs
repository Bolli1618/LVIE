use std::collections::HashMap;

use image::RgbImage;
use LVIElib::hsl::HslImage;
use LVIElib::matrix::convolution::multithreadded::apply_convolution;
use LVIElib::Matrix;

pub struct Filters {}

impl Filters {
    #[allow(dead_code)]
    pub fn GaussianBlur() {}

    pub fn BoxBlur(sigma: u32) -> Matrix<f32> {
        let mut kernel: Vec<f32> = Vec::new();
        let avg: f32 = 1f32 / (sigma.pow(2) as f32);
        for _ in 0..sigma {
            for _ in 0..sigma {
                kernel.push(avg);
            }
        }
        let size = sigma as usize;
        return Matrix::new(kernel, size, size);
    }
}

fn convert_to_hsl(img: &RgbImage) -> HslImage {
    let hsl_img = HslImage::new(img.width(), img.height());
    HslImage::new(img.width(), img.height())
}

pub fn apply_filter(
    img: RgbImage,
    kernel: &mut Matrix<f32>,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    kernel.pad(width as usize, height as usize, 0.0);

    let matrix = Matrix::new(img.into_raw(), height as usize, 3 * width as usize);

    let convolved = apply_convolution(matrix, &kernel);

    image::RgbImage::from_raw(width, height, convolved.get_content().clone()).unwrap()
}

pub fn build_low_res_preview(img: RgbImage) -> RgbImage {
    let resized: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> = image::imageops::resize(
        &img,
        img.width() / 3,
        img.height() / 3,
        image::imageops::Nearest,
    );

    resized
}

pub fn collect_histogram_data(img: &RgbImage) -> [HashMap<u8, u32>; 3] {
    let mut r: HashMap<u8, u32> = HashMap::new();

    for n in 0u8..=u8::MAX {
        r.insert(n, 032);
    }

    let mut g = r.clone();
    let mut b = r.clone();

    for pixel in img.pixels() {
        *r.get_mut(&pixel.0[0]).unwrap() += 1;
        *g.get_mut(&pixel.0[1]).unwrap() += 1;
        *b.get_mut(&pixel.0[2]).unwrap() += 1;
    }

    [r, g, b]
}