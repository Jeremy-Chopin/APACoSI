import numpy as np
from matplotlib import pyplot as plt
import PIL.Image as Image

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def plot_bars(classes, res_1, res_2, label_1, label_2, index, image_gray, image_rgb, gt_gray, corrected):

    x = np.arange(len(classes))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res_1, width, label=label_1)
    rects2 = ax.bar(x + width/2, res_2, width, label=label_2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and type')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig("Results_10/" + str(index) + "_iou.png")
    plt.close()

    fig = plt.figure("Comparaison des résultats")
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)
    plt.subplot(1,4,1); plt.title("Image"); plt.imshow(image_rgb); plt.axis('off')
    plt.subplot(1,4,2); plt.title("Groundtruth"); plt.imshow(gt_gray); plt.axis('off')
    plt.subplot(1,4,3); plt.title("CNN"); plt.imshow(image_gray); plt.axis('off')
    plt.subplot(1,4,4); plt.title("CNN + QAP"); plt.imshow(corrected); plt.axis('off')
    plt.savefig("Results_10/" + str(index) + "_comp.png")
    plt.close()
    #plt.show()