
import os
import numpy as np
from PIL import Image
import cv2

if __name__ == '__main__':
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('test_video_arcane.mp4', fourcc, 20.0, (1920,  1080))
    save_dir = 'old_tree_x1'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cap = cv2.VideoCapture('old_tree.mp4')
    gen_frames = []
    n = 0
    scale = 8
    while cap.isOpened():
        ret, frame = cap.read()
        if not isinstance(frame, np.ndarray):
            break
        H, W, C = frame.shape
        n += 1
        # gen_frames.append(frame)
        # if H != 1080 or W != 1920:
            # frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        frame = frame[(H - 1080)//2:(H + 1080)//2, (W - 1920)//2:(W + 1920)//2, :]
        # frame = cv2.resize(frame, (W//scale, H//scale), interpolation=cv2.INTER_CUBIC)
        # cv2.putText(frame, str(n), (10, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        # cv2.imshow('frame', frame)
        # if n <= 800:
            # continue
        # if n > 1200:
            # break
        # if cv2.waitKey(1) == ord('q'):
            # break
        print('{}\r'.format(n), end='')
        cv2.imwrite(os.path.join(save_dir, '{:08d}.png'.format(n)), frame)

    cap.release()
    cv2.destroyAllWindows()

    # print(n)
    # with get_writer('fungtsun_confuse.gif', mode="I", fps=20) as writer:
        # for i in range(n):
            # if i > 4:
                # out.write(gen_frames[i])
                # writer.append_data(gen_frames[i][:,:,::-1])
    