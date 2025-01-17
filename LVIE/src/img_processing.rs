use std::collections::HashMap;
use std::hash::Hash;
use image::{ImageBuffer, Pixel, Primitive, Rgb, RgbImage, Rgba, RgbaImage};
use LVIElib::matrix::{
    convolution::laplacian_of_gaussian, convolution::multithreadded::apply_convolution, Matrix,
};
use LVIElib::utils::{convert_hsl_to_rgb, convert_rgb_to_hsl, convert_hsla_to_rgba, convert_rgba_to_hsla};

use LVIElib::oklab::{Oklab, OklabImage, Oklaba, OklabaImage};

use LVIElib::matrix::convolution::convolve;

use num_traits::{NumCast, ToPrimitive};

pub struct Filters {}

impl Filters {
    #[allow(dead_code)]
    pub fn GaussianBlur(sigma: u32, size: (u32, u32)) -> Matrix<f32> {
        laplacian_of_gaussian(sigma as f32, size.0 as usize, size.1 as usize)
    }

    #[allow(dead_code)]
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

#[allow(dead_code)]
pub fn apply_filter(
    img: &RgbImage,
    kernel: &mut Matrix<f32>,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    kernel.pad(width as usize, height as usize, 0.0);

    let matrix = Matrix::new(img.to_vec(), height as usize, 3 * width as usize);

    let convolved = apply_convolution(matrix, &kernel);

    image::RgbImage::from_raw(width, height, convolved.get_content().clone()).unwrap()
}

pub fn build_low_res_preview<P>(img: &ImageBuffer<P, Vec<P::Subpixel>>, nwidth: u32, nheight: u32) -> ImageBuffer<P, Vec<P::Subpixel>> 
where P: Pixel + 'static, <P as image::Pixel>::Subpixel: 'static 
{
    let resized = image::imageops::resize(
        img,
        nwidth,
        nheight,
        image::imageops::Nearest,
    );

    resized
}

pub trait Max {
    const MAX: Self;
}

impl Max for f32 {
    const MAX: Self =  f32::MAX;
}

impl Max for u8 {
    const MAX: Self =  u8::MAX;
}

impl Max for u16 {
    const MAX: Self =  u16::MAX;
}

pub fn collect_histogram_data<P>(img: &ImageBuffer<P, Vec<P::Subpixel>>) -> [HashMap<P::Subpixel, u32>; 3] 
where 
    P: Pixel, P::Subpixel: Primitive + std::cmp::Eq + Hash + Max, 
    std::ops::RangeInclusive<P::Subpixel>: IntoIterator,
    <std::ops::RangeInclusive<<P as Pixel>::Subpixel> as IntoIterator>::Item: ToPrimitive
{
    let mut r: HashMap<P::Subpixel, u32> = HashMap::new();

    for n in NumCast::from(0).unwrap()..=P::Subpixel::MAX {
        r.insert(NumCast::from(n).unwrap(), 0u32);
    }

    let mut g = r.clone();
    let mut b = r.clone();

    for pixel in img.pixels() {
        let channels = pixel.channels();
        *r.get_mut(&channels[0]).unwrap() += 1;
        *g.get_mut(&channels[1]).unwrap() += 1;
        *b.get_mut(&channels[2]).unwrap() += 1;
    }

    [r, g, b]
}

use LVIElib::utils::norm_range_f32;

pub fn saturate(img: &RgbImage, value: f32) -> RgbImage {
    let mut hsl_image = convert_rgb_to_hsl(img);
    for (_, _, pixel) in hsl_image.enumerate_pixels_mut() {
        *pixel.saturation_mut() = norm_range_f32(0.0..=1.0, *pixel.saturation() + value / 2f32);
    }
    convert_hsl_to_rgb(&hsl_image)
}

pub fn saturate_rgba(img: &RgbaImage, value: f32) -> RgbaImage {
    let mut hsl_image = convert_rgba_to_hsla(img);
    for (_, _, pixel) in hsl_image.enumerate_pixels_mut() {
        *pixel.saturation_mut() = norm_range_f32(0.0..=1.0, *pixel.saturation() + value / 2f32);
    }
    convert_hsla_to_rgba(&hsl_image)
}

pub fn sharpen(img: &RgbImage, value: f32, size: usize) -> RgbImage {
    let (mut vl, mut va, mut vb) = (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new());
    let mut oklab_image = OklabImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        let ok_pixel = Oklab::from(*pixel);
        let channels = ok_pixel.channels();

        vl.push(channels[0]);
        va.push(channels[1]);
        vb.push(channels[2]);

        oklab_image.put_pixel(x, y, ok_pixel);
    }

    let l_matrix = Matrix::new(vl, img.height() as usize, img.width() as usize);
    let kernel = laplacian_of_gaussian(value, size, size);

    let gradient = convolve(&l_matrix, &kernel);

    let out_l = (l_matrix - gradient).unwrap();

    vl = out_l.get_content().to_owned();

    let mut out: image::ImageBuffer<Rgb<u8>, Vec<u8>> = RgbImage::new(img.width(), img.height());

    let width = img.width();

    for i in 0..vl.len() {
        out.put_pixel(
            i as u32 % width,
            i as u32 / width,
            Rgb::<u8>::from(Oklab::from_components([vl[i], va[i], vb[i]])),
        );
    }

    out
}

pub fn sharpen_rgba(img: &RgbaImage, value: f32, size: usize) -> RgbaImage {
    let (mut vl, mut va, mut vb, mut valpha) = (
        Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new()
    );
    let mut oklab_image = OklabaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        let ok_pixel = Oklaba::from(*pixel);
        let channels = ok_pixel.channels();

        vl.push(channels[0]);
        va.push(channels[1]);
        vb.push(channels[2]);
        valpha.push(channels[3]);

        oklab_image.put_pixel(x, y, ok_pixel);
    }

    let l_matrix = Matrix::new(vl, img.height() as usize, img.width() as usize);
    let kernel = laplacian_of_gaussian(value, size, size);

    let gradient = convolve(&l_matrix, &kernel);

    let out_l = (l_matrix - gradient).unwrap();

    vl = out_l.get_content().to_owned();

    let mut out: image::ImageBuffer<Rgba<u8>, Vec<u8>> = RgbaImage::new(img.width(), img.height());

    let width = img.width();

    for i in 0..vl.len() {
        out.put_pixel(
            i as u32 % width,
            i as u32 / width,
            Rgba::<u8>::from(Oklaba::from_components([vl[i], va[i], vb[i], valpha[i]])),
        );
    }

    out
}

pub fn crop<P: Pixel>(
    img: &ImageBuffer<P, Vec<P::Subpixel>>, 
    x: u32, y:u32, 
    new_width: u32, new_height:u32) -> ImageBuffer<P, Vec<P::Subpixel>>
where <P as image::Pixel>::Subpixel: std::fmt::Debug
{
    let mut out: ImageBuffer<P, Vec<P::Subpixel>> = ImageBuffer::new(new_width, new_height);

    for ny in 0..new_height {
        for nx in 0..new_width {
            out.put_pixel(nx, ny, img.get_pixel(nx + x, ny + y).clone());
        }
    }

    out
}

