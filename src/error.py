from __future__ import print_function
import h5py
import numpy as np
import argparse

#### print the average error saved in the sample file###
parse = argparse.ArgumentParser()
parse.add_argument("--dataset",help="choose a dataset for training",default="human3.6m",type=str)
args = parse.parse_args()


path='/home/lichen/action_prediction/human-motion-prediction-private/samples/'
if args.dataset=='human3.6m':
    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]
else:
    actions = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running",
               "soccer", "walking", "washwindow"]
time = [1,3,7,9,13,24]
errors=[]

print()
print("{0: <16} |".format("milliseconds"), end="")
for ms in [80, 160, 320, 400, 560, 1000]:
    print(" {0:5d} |".format(ms), end="")
print()

with h5py.File(path+'samplefilename', 'r' ) as h5f:
    for action in actions:
        error=h5f['mean_{}_error'.format(action)][:]
        print("{0: <16} |".format(action), end="")
        for ms in time:
            print(" {0:.3f} |".format(error[ms]), end="")
        print()
        errors.append(error)
errors=np.array(errors)
error_ave=np.mean(errors,axis=0)
print("{0: <16} |".format("average"), end="")
for ms in time:
    print(" {0:.3f} |".format(error_ave[ms]), end="")
print()


# errors_paper = []
# print()
# print("{0: <16} |".format("milliseconds"), end="")
# for ms in [80, 160, 320, 400, 560, 1000]:
#     print(" {0:5d} |".format(ms), end="")
# print()
#
# # with h5py.File(path_paper+'allunsupsamples.h5', 'r' ) as h5f:
# with h5py.File(path_paper+'allunsupsamples.h5','r')as h5f:
#     for action in actions:
#         error=h5f['mean_{}_error'.format(action)][:]
#         errors_paper.append(error)
#     errors_paper = np.array(errors_paper)
#     # error_diff = errors_paper - errors
#     error_ave_paper = np.mean(errors_paper, axis=0)
#     # error_ave_diff = error_ave_paper - error_ave
# for i in range(len(actions)):
#     action=actions[i]
#     print("{0: <16} |".format(action), end="")
#     for ms in time:
#         print(" {0:.3f} |".format(errors_paper[i,ms]), end="")
#     print()
#
# print("{0: <16} |".format("average"), end="")
# for ms in time:
#     print(" {0:.3f} |".format(error_ave_paper[ms]), end="")
# print()







