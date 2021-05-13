#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>

torch::Tensor positions = torch::tensor({{-9.4000e-04,  3.0000e-05, -0.0000e+00},
				    { 1.5340e-01, -7.4000e-04,  0.0000e+00},
				    { 2.1526e-01, -8.2700e-02, -1.4310e-01},
				    { 2.1526e-01, -8.2700e-02,  1.4310e-01},
				    {-3.7280e-02,  2.1000e-04,  1.0277e-01},
				    {-3.6830e-02,  8.9200e-02, -5.1380e-02},
				    {-3.7720e-02, -8.8790e-02, -5.1380e-02},
				    { 1.7351e-01,  1.0639e-01, -0.0000e+00}},
  torch::TensorOptions().requires_grad(true).dtype(torch::kDouble));

torch::Tensor signals = torch::tensor({{-0.1121,  1.9069,  0.1078},
				       { 0.1551,  1.9069,  0.1078},
				       {-0.1614,  1.9452,  0.2638},
				       {-0.1614,  1.9452,  0.2638},
				       { 0.0620,  1.4593,  0.0208},
				       { 0.0620,  1.4593,  0.0208},
				       { 0.0620,  1.4593,  0.0208},
				       { 0.0927,  1.2593,  0.0208}},
  torch::TensorOptions().requires_grad(true).dtype(torch::kDouble));

int main(int argc, const char* argv[]) {

  // Get the path of model
  if (argc != 2) {
    std::cerr << "usage: test_model <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module model;
  torch::Device device(torch::kCPU);
  try {
	// Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(argv[1]);
    std::cout << "successfully loaded the model\n";
  }

  catch (const c10::Error& e) {
	std::cerr << "error loading the model\n";
	return -1;
  }

   // Set the model properties
   model.to(device);
   model.to(torch::kDouble);

   // Define the input variables
   std::vector<torch::jit::IValue> inputs;

   inputs.push_back(positions);
   inputs.push_back(signals);

   // Run the model
   torch::Tensor output = model.forward(inputs).toTensor();

   // Test the gradients
   torch::Tensor target = torch::rand_like(output);
   torch::Tensor loss = torch::mse_loss(output, target);
   loss.backward();
   std::cout << "Positions grad are\n" << positions.grad() << std::endl;
   std::cout << "Signals grad are\n" << signals.grad() << std::endl;
 }
