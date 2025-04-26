# cs4476-project-1-convolution-and-hybrid-images-solved
**TO GET THIS SOLUTION VISIT:** [CS4476 Project 1: Convolution and Hybrid Images Solved](https://www.ankitcodinghub.com/product/cs4476-project-1-convolution-and-hybrid-images-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;69808&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;4&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (4 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS4476 Project 1: Convolution and Hybrid Images  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (4 votes)    </div>
    </div>
<h1>Logistics</h1>
<ul>
<li>Due: Check <a href="https://canvas.gatech.edu/">Canvas</a> for up to date information.</li>
<li>Project materials including report template: <a href="https://github.gatech.edu/cs4476/project-1">Project</a> <a href="https://github.gatech.edu/cs4476/project-1">1</a></li>
<li>Hand-in: <a href="https://www.gradescope.com/">Gradescope</a></li>
<li>Required files: &lt;your_gt_username&gt;.zip, &lt;your_gt_username&gt;_project-1.pdf</li>
</ul>
<img data-recalc-dims="1" decoding="async" class="aligncenter lazyloading" data-src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2021/09/667.png?w=980&amp;ssl=1" src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2021/09/667.png?w=980&amp;ssl=1">

<p style="text-align: center;">Figure 1: Look at the image from very close, then from far away.

<h1>Overview</h1>
The goal of this assignment is to write an image filtering function and use it to create hybrid images using a simplified version of the SIGGRAPH 2006 <a href="http://olivalab.mit.edu/publications/OlivaTorralb_Hybrid_Siggraph06.pdf">paper</a> by Oliva, Torralba, and Schyns. <em>Hybrid images </em>are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances.

This project is intended to familiarize you with Python, PyTorch, and image filtering. Once you have created an image filtering function, it is relatively straightforward to construct hybrid images. If you don’t already know Python, you may find <a href="https://docs.python.org/3/tutorial/">this resource</a> helpful. If you are more familiar with MATLAB, <a href="http://mathesaurus.sourceforge.net/matlab-numpy.html">this guide </a>is very helpful. If you’re unfamiliar with PyTorch, the <a href="https://pytorch.org/tutorials/">tutorials</a> from the official website are useful.

<strong>Setup</strong>

<ol>
<li>Found in the README <a href="https://github.gatech.edu/cs4476/project-1">GitHub</a></li>
</ol>
<h1>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Part 1: NumPy</h1>
<h2>1.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Gaussian kernels</h2>
Gaussian filters are used for blurring images. You will first implement create_Gaussian_kernel_1D(), a function that creates a 1D Gaussian vector according to two parameters: the kernel size (length of the 1D vector) and <em>σ</em>, the standard deviation of the Gaussian. The vector should have values populated from evaluating the 1D Gaussian pdf at each coordinate. The 1D Gaussian is defined as:

Next, you will implement create_Gaussian_kernel_2D(), which creates a 2-dimensional Gaussian kernel according to a free parameter, <em>cutoff frequency</em>, which controls how much low frequency to leave in the image. Choosing an appropriate cutoff frequency value is an important step for later in the project when you create hybrid images. We recommend that you implement create_Gaussian_kernel_2D() by creating a 2D Gaussian kernel as the outer product of two 1D Gaussians, which you have now already implemented in create_Gaussian_kernel_1D(). This is possible because the 2D Gaussian filter is <em>separable </em>(think about how <em>e</em><sup>(<em>x</em>+<em>y</em>) </sup>= <em>e<sup>x </sup></em>· <em>e<sup>y</sup></em>). The multivariate Gaussian function is defined as:

where <em>n </em>is equal to the dimension of <em>x</em>, <em>µ </em>is the mean coordinate (where the Gaussian has peak value), and Σ is the covariance matrix.

You will use the value of cutoff frequency to define the size, mean, and variance of the Gaussian kernel. Specifically, the kernel <em>G </em>should be size (<em>k,k</em>) where <em>k </em>= 4 · cutoff frequency + 1, have peak value at

, standard deviation <em>σ </em>= cutoff frequency, and values that sum to 1, i.e., <sup>P</sup><em><sub>ij </sub>G<sub>ij </sub></em>= 1. If your kernel doesn’t sum to 1, you can normalize it as a postprocess by dividing each value by the sum of the kernel.

<h2>1.2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Image Filtering</h2>
Image filtering (or convolution) is a fundamental image processing tool. See chapter 3.2 of Szeliski and the lecture materials to learn about image filtering (specifically linear filtering). You will be writing your own function to implement image filtering from scratch. More specifically, you will implement my_conv2d_numpy() which imitates the filter2D() function in the OpenCV library. As specified in part1.py, your filtering algorithm must: (1) support grayscale and color images, (2) support arbitrarily-shaped filters, as long as both dimensions are odd (e.g., 7 × 9 filters, but not 4 × 5 filters), (3) pad the input image with zeros, and (4) return a filtered image which is the same resolution as the input image. We have provided an iPython notebook, project-1.ipynb and some unit tests (which are called in the notebook) to help you debug your image filtering algorithm. Note that there is a time limit of 5 minutes for a single call to my_conv2d_numpy(), so try to optimize your implementation if it goes over.

<h2>1.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Hybrid Images</h2>
A hybrid image is the sum of a low-pass filtered version of one image and a high-pass filtered version of another image. As mentioned above, <em>cutoff frequency </em>controls how much high frequency to leave in one image and how much low frequency to leave in the other image. In cutoff_frequencies.txt, we provide a default value of 7 for each pair of images (the value of line <em>i </em>corresponds to the cutoff frequency value for the <em>i</em>-th image pair). You should replace these values with the ones you find work best for each image pair. In the paper it is suggested to use two cutoff frequencies (one tuned for each image), and you are free to try that as well. In the starter code, the cutoff frequency is controlled by changing the standard deviation of the Gaussian filter used in constructing the hybrid images. You will first implement create_hybrid_image() according to the starter code in part1.py. Your function will call my_conv2d_numpy() using the kernel generated from create_Gaussian_kernel() to create low and high frequency images, and then combine them into a hybrid image.

<h1>2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Part 2: PyTorch</h1>
<h2>2.1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dataloader</h2>
You will now implement creating hybrid images again but using PyTorch. The HybridImageDataset class in part2_datasets.py will create tuples using pairs of images with a corresponding cutoff frequency (which you should have found from experimenting in Part 1). The image paths will be loaded from data/ using make_dataset() and the cutoff frequencies from cutoff_frequencies.txt using get_cutoff_frequencies(). Additionally, you will implement __len__(), which returns the number of image pairs, and __getitem__(), which returns the i-th tuple. Refer to <a href="https://pytorch.org/tutorials/beginner/data_loading_tutorial.html">this tutorial</a> for additional information on data loading &amp; processing.

<h2>2.2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Model</h2>
Next, you will implement the HybridImageModel class in part2_models.py. Instead of using your implementation of my_conv2d_numpy() to get the low and high frequencies from a pair of images, low_pass() should use the 2D convolution operator from torch.nn.functional to apply a low pass filter to a given image. You will have to implement get_kernel() which calls your create_Gaussian_kernel() function from part1.py for each pair of images using the cutoff frequencies as specified in cutoff_frequencies.txt, and reshape it to the appropriate dimensions for PyTorch. Then, similar to create_hybrid_image() from part1.py, forward() will call get_kernel() and low_pass() to create the low and high frequency images, and combine them into a hybrid image. Refer to <a href="https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html">this tutorial</a> for additional information on defining neural networks using PyTorch.

You will compare the runtimes of your hybrid image implementations from Parts 1 &amp; 2.

<h1>3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Part 3: Understanding input/output shapes in PyTorch</h1>
You will now implement my_conv2d_pytorch() in part3.py using the same 2D convolution operator from torch.nn.functional used in low_pass().

Before we proceed, here are two quick definitions of terms we’ll use often when describing convolution:

<ul>
<li><strong>Stride</strong>: When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice) then the filters jump 2 pixels at a time as we slide them around.</li>
<li><strong>Padding</strong>: The amount of pixels added to an image when it is being convolved with a the kernel. Padding can help prevent an image from shrinking during the convolution operation.</li>
</ul>
Unlike my_conv2d_numpy() from part1.py, the shape of your output does not necessarily have to be the same as the input image. Instead, given an input image of shape (1<em>,d</em><sub>1</sub><em>,h</em><sub>1</sub><em>,w</em><sub>1</sub>) and kernel of shape (), your output will be of shape (1<em>,d</em><sub>2</sub><em>,h</em><sub>2</sub><em>,w</em><sub>2</sub>) where <em>g </em>is the number of groups,&nbsp; + 1, and

+ 1, and <em>p </em>and <em>s </em>are padding and stride, respectively.

Think about <em>why </em>the equations for output width <em>w</em><sub>2 </sub>and output height <em>h</em><sub>2 </sub>are true – try sketching out a 5 × 5 grid, and seeing how many places you can place a 3 × 3 square within the grid with stride 1. What about with stride 2? Does your finding match what the equation states?

We demonstrate the effect of the value of the groups parameter on a simple example with an input image of shape (1<em>,</em>2<em>,</em>3<em>,</em>3) and a kernel of shape (4<em>,</em>1<em>,</em>3<em>,</em>3):

<img data-recalc-dims="1" decoding="async" class="aligncenter lazyload" data-src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2021/09/159.png?w=980&amp;ssl=1" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

<p style="text-align: center;">Figure 2: Visualization of a simple example using groups=2.

<h1>4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Writeup</h1>
For this project (and all other projects), you must do a project report using the template slides provided to you. Do <em>not </em>change the order of the slides or remove any slides, as this will affect the grading process on Gradescope and you will be deducted points. In the report you will describe your algorithm and any decisions you made to write your algorithm a particular way. Then you will show and discuss the results of your algorithm. The template slides provide guidance for what you should include in your report. A good writeup doesn’t just show results–it tries to draw some conclusions from the experiments. You must convert the slide deck into a PDF for your submission.

If you choose to do anything extra, add slides <em>after the slides given in the template deck </em>to describe your implementation, results, and analysis. Adding slides in between the report template will cause issues with Gradescope, and you will be deducted points. You will not receive full credit for your extra credit implementations if they are not described adequately in your writeup.

<h1>Data</h1>
We provide you with 5 pairs of aligned images which can be merged reasonably well into hybrid images. The alignment is super important because it affects the perceptual grouping (read the paper for details). We encourage you to create additional examples (e.g., change of expression, morph between different objects, change over time, etc.).

For the example shown in Figure 1, the two original images look like this:

The low-pass (blurred) and high-pass version of these images look like this:

The high frequency image in Figure 4b is actually zero-mean with negative values, so it is visualized by adding 0.5. In the resulting visualization, bright values are positive and dark values are negative.

Adding the high and low frequencies together (Figures 4b and 4a, respectively) gives you the image in

<img data-recalc-dims="1" decoding="async" class="aligncenter lazyload" data-src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2021/09/355.png?w=980&amp;ssl=1" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

<p style="text-align: center;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (a) Dog&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) Cat

Figure 3

<img data-recalc-dims="1" decoding="async" class="aligncenter lazyload" data-src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2021/09/841.png?w=980&amp;ssl=1" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

<p style="text-align: center;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (a) Low frequencies of dog image.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) High frequencies of cat image.

Figure 4

Figure 1. If you’re having trouble seeing the multiple interpretations of the image, a useful way to visualize the effect is by progressively downsampling the hybrid image, as done in Figure 5. The starter code provides a function, vis_image_scales_numpy() in utils.py, which can be used to save and display such visualizations.

<h2>Potentially useful NumPy (Python library) functions</h2>
np.pad(), which does many kinds of image padding for you, np.clip(), which “clips” out any values in an array outside of a specified range, np.sum() and np.multiply(), which makes it efficient to do the convolution (dot product) between the filter and windows of the image. Documentation for NumPy can be found <a href="https://docs.scipy.org/doc/numpy/">here</a> or by Googling the function in question.

<h2>Forbidden functions</h2>
(You can use these for testing, but not in your final code). Anything that takes care of the filter operation or creates a 2D Gaussian kernel directly for you is forbidden. If it feels like you’re sidestepping the work, then it’s probably not allowed. Ask the TAs if you have any doubts.

<img data-recalc-dims="1" decoding="async" class="aligncenter lazyload" data-src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2021/09/150.png?w=980&amp;ssl=1" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

<p style="text-align: center;">Figure 5

<h1>Testing</h1>
We have provided a set of tests for you to evaluate your implementation. We have included tests inside project-1.ipynb so you can check your progress as you implement each section. When you’re done with the entire project, you can call additional tests by running pytest tests inside the root directory of the project, as well as checking against the tests on Gradescope. <em>Your grade on the coding portion of the project will be further evaluated with a set of tests not provided to you.</em>

<h1>Bells &amp; whistles (extra points)</h1>
For later projects there will be more concrete extra credit suggestions. It is possible to get extra credit for this project as well if you come up with some clever extensions which impress the TAs. If you choose to do extra credit, you should add slides <em>at the end </em>of your report further explaining your implementation, results, and analysis. You will not be awarded credit if this is missing from your submission.

<strong>Rubric</strong>

See <a href="https://github.gatech.edu/cs4476/project-1">Project 1 README</a><a href="https://github.gatech.edu/cs4476/project-1">.</a>

<strong>Submission</strong>

See <a href="https://github.gatech.edu/cs4476/project-1">Project 1 README</a><a href="https://github.gatech.edu/cs4476/project-1">.</a>

<h1>Credits</h1>
Assignment developed by James Hays, Cusuh Ham, John Lambert, Vijay Upadhya, Samarth Brahmbhatt, and Frank Dellaert, based on a similar project by Derek Hoiem.
