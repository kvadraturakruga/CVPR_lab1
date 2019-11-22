import cv2, os, time
import numpy as np
def kaze_match(images_path, good_image_path, pref, f, pos):
    images = []
    for file in os.listdir(images_path):
        if file.endswith(".jpg"):
            images.append(cv2.imread(images_path+file))

    gray_images = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray);

    good_image = cv2.imread(good_image_path)
    good_gray_image = cv2.cvtColor(good_image, cv2.COLOR_BGR2GRAY)

    detector = cv2.KAZE_create()

    results = []
    for image in gray_images:
        (kps, desc) = detector.detectAndCompute(image, None)
        results.append((kps,desc,image))

    (kps_good, descs_good) = detector.detectAndCompute(good_gray_image, None)

    i = 0
    
    for (kps, desc, image) in results:
        f.write(pref+" Image "+str(pos)+"\n")
        f.write("keypoints: {}, descriptors: {}".format(len(kps), desc.shape)+"\n")
        print("keypoints: {}, descriptors: {}".format(len(kps), desc.shape))
        print("i: ",i)
        start_time = time.time()

        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=10)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(desc,descs_good,k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(image, kps, good_gray_image, kps_good, good, None, flags=2)
        cv2.imwrite("../results_kaze/"+pref+"res"+str(pos)+".jpg", img3)
        #cv2.imwrite("../results_kaze2/"+pref+"res"+str(pos)+".jpg", img3)

        f.write("Time: {}\n".format(time.time()-start_time))
        i = i + 1
        pos = pos +1


def orb_match(images_path, good_image_path, pref, f, pos):
    images = []
    for file in os.listdir(images_path):
        if file.endswith(".jpg"):
            images.append(cv2.imread(images_path+file))

    gray_images = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray);

    good_image = cv2.imread(good_image_path)
    good_gray_image = cv2.cvtColor(good_image, cv2.COLOR_BGR2GRAY)

    detector = cv2.ORB_create()
    results = []
    for image in gray_images:
        (kps,desc) = detector.detectAndCompute(image, None)
       
        results.append((kps,desc,image))

    (kps_good,descs_good) = detector.detectAndCompute(good_gray_image, None)
   
    i = 0
    
    for (kps, desc, image) in results:
        f.write(pref+" Image "+str(pos)+"\n")
        f.write("keypoints: {}, descriptors: {}".format(len(kps), desc.shape)+"\n")
        print("keypoints: {}, descriptors: {}".format(len(kps), desc.shape))
        print("i: ",i)
        start_time = time.time()

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
        matches = bf.match(desc,descs_good)

# Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        

        good = np.array([])
       
        img3 = cv2.drawMatches(image, kps, good_gray_image, kps_good, matches[:20], good, flags=2)
        cv2.imwrite("../results_orb/"+pref+"res"+str(pos)+".jpg", img3)
        #cv2.imwrite("../results_orb2/"+pref+"res"+str(pos)+".jpg", img3)
        
        f.write("Time: {}\n".format(time.time()-start_time))
        i = i + 1
        pos = pos +1
    

def main():
    good_image_dir = "../images good/"
    bad_image_dir = "../images bad/"
    good_image = "../images good/img1.jpg"
    #good_image_dir = "../images good2/"
    #bad_image_dir = "../images bad2/"
    #good_image = "../images good2/img1.jpg"
    ####
    f1 = open("results_kaze.txt", "w")
    f1.write("Results:\n")
    f1.close()
    f1 = open("results_kaze.txt", "a")
    f2 = open("results_orb.txt", "w")
    f2.write("Results:\n")
    f2.close()
    f2 = open("results_orb.txt", "a")
    
#####
    #f1 = open("results_kaze2.txt", "w")
    #f1.write("Results:\n")
    #f1.close()
    #f1 = open("results_kaze2.txt", "a")
    #f2 = open("results_orb2.txt", "w")
    #f2.write("Results:\n")
    #f2.close()
    #f2 = open("results_orb2.txt", "a")
    #####
    
    kaze_match(good_image_dir,good_image,"good",f1, pos =0)
    orb_match(good_image_dir,good_image,"good",f2, pos = 0)
    print("good matches first time")
    good_image_dir = "../images good12/"
    #good_image_dir = "../images good22/"
    kaze_match(good_image_dir,good_image,"good",f1,pos=100)
    orb_match(good_image_dir,good_image,"good",f2, pos = 100)

    good_image_dir = "../images good13/"
    #good_image_dir = "../images good23/"

    kaze_match(good_image_dir,good_image,"good",f1,pos=200)
    orb_match(good_image_dir,good_image,"good",f2,pos=200)

    good_image_dir = "../images good14/"
    #good_image_dir = "../images good24/"

    kaze_match(good_image_dir,good_image,"good",f1,pos=300)
    orb_match(good_image_dir,good_image,"good",f2,pos=300)

    kaze_match(bad_image_dir,good_image,"bad",f1, pos =0)
    orb_match(bad_image_dir,good_image,"bad",f2, pos = 0)
    print("already matches")
main()