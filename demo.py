
from src import model
from src import util
from src.body import Body
from src.hand import Hand
import argparse
import glob
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="Select the image path")
    parser.add_argument("--results_dir")
    args = parser.parse_args()
    return args


def main(args):

    file_names = glob.glob(args.input_folder + "/Screen*.jpg")
    body_estimation = Body('model/body_pose_model.pth')

    for each in file_names:
        test_image = each
        file_name_with_format = test_image.split("/")[-1]
        file_name = file_name_with_format.split(".")[0]
        oriImg = cv2.imread(test_image)  # B,G,R order
        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        print(os.path.join(args.results_dir, file_name + ".png"))
        cv2.imwrite(os.path.join(args.results_dir, file_name + ".png"), canvas)
        final_result = candidate[0]
        with open(os.path.join(args.results_dir, file_name + ".json", "w")) as out_file:
            json.dump({"result": final_result}, out_file)


if __name__ == '__main__':
    main(parse_args())

