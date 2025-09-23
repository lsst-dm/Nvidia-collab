from lsst.daf.butler import Butler

from lsst.afw.image import ImageD
from lsst.ip.diffim.makeKernelBasisList import makeKernelBasisList
from lsst.ip.diffim.makeKernel import MakeKernelConfig, MakeKernelTask

import numpy as np

import numpy as np
from scipy.signal import fftconvolve

def apply_kernel(input_image, basis_functions, basis_radius, spatial_order, basis_coefficients):
    input_shape = input_image.shape
    output_shape = (input_shape[0] - 2 * basis_radius, input_shape[1] - 2 * basis_radius)
    x_mid = input_shape[1] / 2
    y_mid = input_shape[0] / 2

    # Convolve each basis function with the input image
    convolved_basis = [fftconvolve(input_image, basis, mode='same') for basis in basis_functions]
    convolved_basis = np.array(convolved_basis)  # Shape: (num_basis, H, W)

    # Create coordinate grids
    y_indices = np.arange(0, input_shape[0])
    x_indices = np.arange(0, input_shape[1])
    y_pos, x_pos = np.meshgrid(x_indices, y_indices)

    # Normalize coordinates
    y_pos_norm = (y_pos - y_mid) / y_mid
    x_pos_norm = (x_pos - x_mid) / x_mid

    # Initialize Chebyshev polynomials
    y_cheb = np.zeros((spatial_order + 1, input_shape[0], input_shape[1]))
    x_cheb = np.zeros((spatial_order + 1, input_shape[0], input_shape[1]))
    y_cheb[0, :, :] = 1.0
    x_cheb[0, :, :] = 1.0

    y_cheb[1, :, :] = y_pos_norm
    x_cheb[1, :, :] = x_pos_norm

    # Compute Chebyshev polynomials using recurrence
    for i in range(2, spatial_order + 1):
        y_cheb[i, :, :] = 2 * y_pos_norm * y_cheb[i - 1, :, :] - y_cheb[i - 2, :, :]
        x_cheb[i, :, :] = 2 * x_pos_norm * x_cheb[i - 1, :, :] - x_cheb[i - 2, :, :]

    # Compute outer product using broadcasting
    # x_cheb is (O+1, H, W), y_cheb is (O+1, H, W)
    # Broadcast to (O+1, O+1, H, W)
    spatial_terms = x_cheb[:, None, :, :] * y_cheb[None, :, :, :]

    # Flatten the outer product for each (x, y) position
    # Resulting shape: ( (O+1)^2, H, W )
    spatial_terms = spatial_terms.reshape(( (spatial_order + 1) ** 2, input_shape[0], input_shape[1] ))

    
    spatial_indices = []
    index = 0
    for i in range((spatial_order + 1)):
        for _ in range(0, spatial_order-i + 1):
            spatial_indices.append(index)
            index += 1
    
    # Apply the kernel: basis * spatial_terms
    # Note: This assumes that `basis_functions` corresponds to all combinations of Chebyshev polynomials
    # For general usage, adjust this part accordingly
    output_array = np.zeros(input_image.shape)
    bc = iter(basis_coefficients)
    for i in range(convolved_basis.shape[0]):
        for j in spatial_indices:
            basis_value = convolved_basis[i]
            output_array += basis_value * spatial_terms[j] * next(bc)

    return output_array[basis_radius:-basis_radius, basis_radius:-basis_radius]


def solve_diff_kernel(x_values, y_values, basis_functions, spatial_order, template_image, target_image):
    """
    Solves for the coefficients of a differential kernel.

    Args:
        x_values (np.ndarray): 1D array of x coordinates.
        y_values (np.ndarray): 1D array of y coordinates.
        basis_functions (np.ndarray): 3D array of basis functions.
        spatial_order (int): The spatial order of the kernel.
        template_image (np.ndarray): 2D array of the template image.
        target_image (np.ndarray): 2D array of the target image.

    Returns:
        dict: A dictionary containing the basis arrays, basis radius, spatial order, and basis coefficients.
    """

    kernel_width = int(basis_functions.shape[1] / 2)
    template_shape = template_image.shape
    x_mid = template_shape[1] // 2
    y_mid = template_shape[0] // 2

    # Initialize Chebyshev polynomials
    y_cheb = np.zeros((spatial_order + 1,))
    x_cheb = np.zeros((spatial_order + 1,))
    y_cheb[0] = 1.0
    x_cheb[0] = 1.0

    # Filter out x and y values that are too close to the bounds
    xy_positions = list(zip(x_values, y_values))
    xy_positions = [
        (x, y)
        for x, y in xy_positions
        if (
            kernel_width < x < template_shape[1] - kernel_width - 2
            and kernel_width < y < template_shape[0] - kernel_width - 2
        )
    ]

    # Calculate number of parameters
    num_parameters = 0
    for i in range(spatial_order + 1):
        for _ in range(spatial_order + 1 - i):
            num_parameters += 1
    num_parameters *= basis_functions.shape[0]

    basis_accumulator = np.zeros((num_parameters, num_parameters))
    target_accumulator = np.zeros(num_parameters)

    for x, y in xy_positions:
        y_start = y - kernel_width
        y_stop = y + kernel_width + 1
        x_start = x - kernel_width
        x_stop = x + kernel_width + 1

        template_view = template_image[y_start:y_stop, x_start:x_stop]
        basis_values = np.sum(basis_functions * template_view, axis=(1,2))
        basis_values_column = basis_values.reshape(basis_values.shape[0], 1)

        poly_y_pos = (y - y_mid) / y_mid
        poly_x_pos = (x - x_mid) / x_mid

        y_cheb[1] = poly_y_pos
        x_cheb[1] = poly_x_pos

        for i in range(2, spatial_order + 1):
            y_cheb[i] = 2 * poly_y_pos * y_cheb[i - 1] - y_cheb[i - 2]
            x_cheb[i] = 2 * poly_x_pos * x_cheb[i - 1] - x_cheb[i - 2]

        y_cheb_row = y_cheb.reshape(1, y_cheb.shape[0])
        x_cheb_column = x_cheb.reshape(x_cheb.shape[0], 1)

        spatial_terms = x_cheb_column @ y_cheb_row
        spatial_term_filtered = []
        for i in range(spatial_terms.shape[0]):
            spatial_term_filtered.extend(spatial_terms[i, :spatial_terms.shape[1] - i].tolist())

        spatial_term_filtered = np.array(spatial_term_filtered)
        
        terms = (basis_values_column @ spatial_term_filtered.reshape((1,len(spatial_term_filtered)))).flatten()
        
        basis_accumulator += np.outer(terms, terms) # Use outer product

        target_value = target_image[(y - kernel_width), (x - kernel_width)]
        target_accumulator += terms * target_value

    try:
        coefficients = np.linalg.solve(basis_accumulator, target_accumulator)
    except np.linalg.LinAlgError:
        print("Singular matrix.  Returning zeros.")
        coefficients = np.zeros_like(target_accumulator)

    return {
        "basis_functions": [arr.copy() for arr in basis_functions],
        "basis_radius": kernel_width,
        "spatial_order": spatial_order,
        "basis_coefficients": coefficients.astype(np.float32),
    }


butler = Butler("$AP_VERIFY_DIR/workspaces/cosmos/repo/", collections="ap_verify-output")

visit = 59150
detector = 58
instrument = "HSC"
band = "g"

science_image = butler.get("preliminary_visit_image", visit=visit, detector=detector, instrument=instrument)

template_image = butler.get("template_detector", visit=visit, instrument=instrument, detector=detector)

science_size = 4.076
template_size = 3.877

config = MakeKernelConfig()
task = MakeKernelTask(config)
kernel_images = makeKernelBasisList(config.kernel.active, science_size, template_size)

ks = [ImageD(21, 21) for i in range(len(kernel_images))]
for ker, k in zip(kernel_images, ks):
    ker.computeImage(k, True)
cand_list = task.makeCandidateList(template_image, science_image,kernel_images[0].getWidth(),None,None)

y_s = []
x_s = []
for f in cand_list:
    y,x = f.getFootprint().spans.indices()
    y_s.extend(y)
    x_s.extend(x)


y_range = np.array(y_s)
x_range = np.array(x_s)

basis_array = np.zeros((len(ks), 21,21), dtype=np.float32)

for i in range(len(ks)):
    basis_array[i] = ks[i].array.astype(np.float32)


t1 = time.time()
new_results = solve_diff_kernel(x_range, y_range, basis_array, 2, template_image.image.array[10:-10, 10:-10].astype(np.float32), science_image.image.array.astype(np.float32))
t2 = time.time()
print(f"the time was{t1-t2}")

convolved = apply_kernel(template_image.image.array[10:-10, 10:-10].astype(np.float32), **new_results)

diff = 
