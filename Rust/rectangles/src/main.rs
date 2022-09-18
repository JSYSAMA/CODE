fn area(width: u32,height: u32) -> u32 {
    width * height
}
fn main() {
    let width = 30;
    let height = 20;
    println!("The area of rectangle is {}.", area(width, height));
}
