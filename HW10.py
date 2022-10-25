# import the opencv library
import numpy as np
import cv2
  
def lucas_kanade_optical_flow(video_device) :
    cap = cv2.VideoCapture(video_device)
    #----------------------------params for ShiTomasi corner detection----------------------------#
    feature_params = dict(  maxCorners = 900,
                            qualityLevel = 0.03,
                            minDistance = 10,
                            blockSize = 50 )
    #--------------------------Parameters for lucas kanade optical flow---------------------------#
    lk_params = dict(   winSize  = (21,21),
                        maxLevel = 3,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #----------------------------------------Create colors----------------------------------------#
    Line = (0,0,255)  # Line
    Head = (0,255,0) # Head
    #---------------------------------------Take first frame--------------------------------------#
    ret, old_frame = cap.read()
    #-----------------------------------------Find corner-----------------------------------------#
    stencil = np.zeros(old_frame.shape).astype(old_frame.dtype)
    myROI = [(720,476), (530,25 ), (169, 25), (0,476)]  # (x, y)
    cv2.fillPoly(stencil, [np.array(myROI)], (255,255,255))
    old_frame = cv2.bitwise_and(old_frame, stencil)
    #--------------Feature detection, Harris corner with Shi-Tomasi response function-------------#
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    #-------------------------------Create a mask image for overlay-------------------------------#
    mask = np.zeros_like(old_frame)
    while cap.isOpened() : 
        ret, frame = cap.read()
        stencil = np.zeros(frame.shape).astype(frame.dtype)
        myROI = [(720,476), (530,25 ), (169, 25), (0,476)]  # (x, y)
        cv2.fillPoly(stencil, [np.array(myROI)], (255,255,255))
        frame = cv2.bitwise_and(frame, stencil)
        #-----------------------------------Working conditions------------------------------------#
        if ret :
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #-------------------------------Calculate optical flow--------------------------------#
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )
            #---------------------------------Select Best points----------------------------------#
            best_new = p1[st == 1]
            best_old = p0[st == 1]
            #----------------------------------Traveline drawing----------------------------------#
            for i, (new, old) in enumerate(zip(best_new, best_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                mask = cv2.line(mask, (a, b), (c, d), Line, 2)
                frame = cv2.circle(frame, (a,b), 5, Head, -5)
            compare_img = cv2.hconcat([frame, mask])
            disp_img = cv2.add(frame, mask)
            cv2.imshow('frame', disp_img)
            key = cv2.waitKey(27) & 0xFF
            if key == 27 or key == ord('q') :
                break
            #-------------------------------------Clear mask--------------------------------------#
            elif key == ord('c') :
                mask = np.zeros_like(old_frame)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            #--------------------Update the previous frame and previous points--------------------#
            else :
                old_gray = frame_gray.copy()
                p0 = best_new.reshape(-1, 1, 2)
        else :
            break

    cap.release()
    cv2.destroyAllWindows()

lucas_kanade_optical_flow("/Users/nk/Documents/IMG/HW10/grandcentral.mp4")

cv2.waitKey(0)
cv2.destroyAllWindows()
#-------------------------------------------End Codeing-------------------------------------------#