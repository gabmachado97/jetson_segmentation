import jetson.inference
import jetson.utils

import argparse
import sys

from segnet_utils import *

# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=175.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 175.0)")
parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

# create buffer manager
buffers = segmentationBuffers(net, opt)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

#rtk input
width = 176.0
height = 144.0

# determine the amount of border pixels (cropping around the center by half)`
crop_border = (height//5,
		height//4)


# compute the ROI as (left, top, right, bottom)
crop_roi = (1.0, crop_border[0], width-1, height - crop_border[1])
#print(crop_roi)


# process frames until user exits
while True:
	# capture the next image
	img_input = input.Capture()

	# allocate buffers for this size image
	buffers.Alloc(img_input.shape, img_input.format)

	# process the segmentation network
	net.Process(img_input, ignore_class=opt.ignore_class)

	# generate the overlay
	if buffers.overlay:
		net.Overlay(buffers.overlay, filter_mode=opt.filter_mode)

	# generate the mask
	if buffers.mask:
		net.Mask(buffers.mask, filter_mode=opt.filter_mode)
		#allocate the output image, with the cropped size
		img_roi = jetson.utils.cudaAllocMapped(width=width-2,
                                         height=height - crop_border[1] - crop_border[0],
                                         format=buffers.mask.format)
		jetson.utils.cudaCrop(buffers.mask, img_roi, crop_roi)
		found_first_cxy = False
		last_cx = 0
		last_cy = 0 
		for y in range(img_roi.height):
			for x in range(img_roi.width):
				pixel = img_roi[y,x]    # returns a tuple, i.e. (r,g,b) for RGB formats or (r,g,b,a) for RGBA formats
				if pixel == (85,85,255) or pixel == (255,170,127) or pixel == (85,170,127):
					img_roi[y,x] = (255,255,255)
					last_cx = x
					last_cy = y
					if found_first_cxy == False:
						first_cx = x
						first_cy = y
						found_first_cxy = True

		if last_cx != 0:
 			cx = (last_cx - first_cx)//2
		if last_cy != 0:
			cy = (last_cy - first_cy)//2
		print("Coordenadas Centro Caminho Navegável: (%d , %s)"%(cx,cy))
		# cudaDrawCircle(input, (cx,cy), radius, (r,g,b,a), output=None)
		#jetson.utils.cudaDrawCircle(img_roi, (cx,cy), 15, (255,0,0,0))



	# composite the images
	if buffers.composite:
		jetson.utils.cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
		jetson.utils.cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)
		jetson.utils.cudaOverlay(img_roi, buffers.composite, buffers.overlay.width, buffers.mask.height)

	# render the output image
	output.Render(buffers.output)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	jetson.utils.cudaDeviceSynchronize()
	net.PrintProfilerTimes()

    # compute segmentation class stats
	if opt.stats:
		buffers.ComputeStats()
    
	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
