use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = flate2::GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}


#[derive(Debug)]
pub struct MnistImage {
    pub image: Array2<f64>,
    pub classification: u8,
}

pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("{}-labels-idx1-ubyte.gz", dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Array2<f64>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        images.push(Array2::from_shape_vec((image_shape, 1), image_data).unwrap());
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let mut ret: Vec<MnistImage> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MnistImage {
            image,
            classification,
        })
    }

    Ok(ret)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn forward_feed(input: &Array1<f64>, weights: &Vec<Array2<f64>>) -> Array1<f64> {
    let mut layer_output = input.to_owned();
    for weight in weights {
        layer_output = layer_output.dot(weight);
        layer_output.mapv_inplace(sigmoid);
    }
    layer_output
}

fn backpropagate(
    input: &Array1<f64>,
    output: &Array1<f64>,
    weights: &Vec<Array2<f64>>,
    learning_rate: f64,
) -> Vec<Array2<f64>> {
    let mut gradients: Vec<Array2<f64>> = vec![Array2::zeros((1, 1)); weights.len()];

    // Compute the output layer gradient
    let output_activation = forward_feed(input, weights);
    let error = output - &output_activation;
    let output_gradient = -2.0 * error * &output_activation.mapv(sigmoid_derivative);

    // Compute the hidden layer gradients
    let mut layer_output = input.to_owned();
    let mut layer_weights = &weights[0];
    let mut layer_gradient = output_gradient.to_owned();
    for (i, weight) in weights.iter().skip(1).enumerate() {
        layer_output = layer_output.t().dot(layer_weights);
        let layer_input = layer_output.mapv(sigmoid_derivative);
        layer_gradient = layer_input * layer_gradient.dot(&layer_weights.t());
        gradients[weights.len() - i - 1] = layer_output.insert_axis(Axis(0)).t().dot(&layer_gradient.insert_axis(Axis(0)));
        layer_weights = weight;
    }

    // Compute the input layer gradient
    layer_output = input.to_owned();
    for weight in weights.iter().rev().skip(1) {
        layer_output = layer_output.dot(&weight.t());
    }
    let input_gradient = layer_output.mapv(sigmoid_derivative) * layer_gradient.dot(&weights[0].t());
    gradients[0] = input.to_owned().insert_axis(Axis(1)).t().dot(&input_gradient.insert_axis(Axis(0)));

    // Apply the gradients to the weights
    for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
        *weight = *weight - learning_rate * gradient;
    }

    gradients
}

fn main() {
    // Define the neural network architecture
    let input_size = 2;
    let hidden_sizes = vec![4, 3];
    let output_size = 10;

    // Initialize the weights randomly
    let mut weights = Vec::new();
    let mut prev_size = input_size;
    for size in hidden_sizes.iter() {
        let weight = Array2::random((prev_size, *size), Uniform::new(-1.0, 1.0));
        weights.push(weight);
        prev_size = *size;
    }
    let weight = Array2::random((prev_size, output_size), Uniform::new(-1.0, 1.0));
    weights.push(weight);

    // Define the training data
    let inputs = Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    let targets = Array2::from_shape_vec((4, 1), vec![0.5, 0.3, 0.2, 0.1]).unwrap();

    // Train the neural network
    let learning_rate = 0.5;
    for _ in 0..1000 {
        for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
            backpropagate(&input.to_owned(), &target.to_owned(), &mut weights, learning_rate);
        }
    }

    // Test the neural network
    for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
        let output = forward_feed(&input.to_owned(), &weights);
        println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, output);
    }

    // Print the weights
    for weight in weights.iter() {
        println!("{:?}", weight);
    }

    // Print the gradients
    for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
        let gradients = backpropagate(&input.to_owned(), &target.to_owned(), &weights, learning_rate);
        for gradient in gradients {
            println!("{:?}", gradient);
        }
    }

    // Print the gradients numerically
    // for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
    //     let mut gradients = Vec::new();
    //     for weight in weights.iter() {
    //         let mut gradient = Array2::zeros(weight.raw_dim());
    //         for (index, _) in weight.indexed_iter() {
    //             let mut weight = weight.to_owned();
    //             weight[index] += 0.0001;
    //             let output1 = forward_feed(&input, &weights);
    //             let error1 = (target - output1).mapv(|x| x.powi(2)).sum();
    //             let output2 = forward_feed(&input, &vec![weight]);
    //             let error2 = (target - output2).mapv(|x| x.powi(2)).sum();
    //             gradient[index] = (error2 - error1) / 0.0001;
    //         }
    //         gradients.push(gradient);
    //     }
    //     for gradient in gradients {
    //         println!("{:?}", gradient);
    //     }
    // }



}
