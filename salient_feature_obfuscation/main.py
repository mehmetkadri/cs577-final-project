import os
import json
import random
import requests
import cv2 as cv
import numpy as np
import face_recognition
from matplotlib import pyplot as plt
import argparse

def organize_metadata():

    if os.path.isfile(json_file):
        print("Metadata is organized. Loading images from json file.")
        return json.load(open(json_file, "r"))

    images = []
    if not os.path.isdir("metadata"):
        print("Metadata folder not found.")
        return images
    elif not os.path.isfile("metadata/facescrub_actors.txt"):
        print("facescrub_actors.txt not found.")
        return images
    elif not os.path.isfile("metadata/facescrub_actresses.txt"):
        print("facescrub_actresses.txt not found.")
        return images
    
    with open("metadata/facescrub_actors.txt", "r") as file:
        cols = file.readline().strip().split("\t")

        images = {}

        for line in file:
            name = line.split("\t")[0]
            if name not in images:
                count = 1
                images[name] = []
            info = {}
            info["id"] = count
            info["image_id"] = line.split("\t")[2]
            info["url"] = line.split("\t")[3]
            info["face_location"] = {
                "x1": line.split("\t")[4].split(",")[0],
                "y1": line.split("\t")[4].split(",")[1],
                "x2": line.split("\t")[4].split(",")[2],
                "y2": line.split("\t")[4].split(",")[3]
            }
            images[name].append(info)
            count += 1

    with open("metadata/facescrub_actresses.txt", "r") as file:
        cols = file.readline().strip().split("\t")

        for index, line in enumerate(file):
            name = line.split("\t")[0]
            if name not in images:
                count = 1
                images[name] = []
            info = {}
            info["id"] = count
            info["image_id"] = line.split("\t")[2]
            info["url"] = line.split("\t")[3]
            info["face_location"] = {
                "x1": line.split("\t")[4].split(",")[0],
                "y1": line.split("\t")[4].split(",")[1],
                "x2": line.split("\t")[4].split(",")[2],
                "y2": line.split("\t")[4].split(",")[3]
            }
            images[name].append(info)
            count += 1

    json.dump(images, open(json_file, "w"), indent=4)

    return images

def download_image(images, max_images=3, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}):
    
    os.chdir("org_images")
    downloaded_images = []

    for name, image_data in images.items():
        # name = "Aaron Eckhart"
        # image_data = images[name] (list of images for Aaron Eckhart)
        
        got_images = True

        if not os.path.isdir(name):
            os.mkdir(name)
        if len(os.listdir(name)) < max_images:
            print("\n", name , " has less than ", max_images, " images. Downloading images...")
            got_images = False
        if got_images:
            files = os.listdir(name)
            for file in files:
                downloaded_images.append(f"{name}/{file.split('.')[0]}")
        if not got_images:
            counter = 0
            for image in image_data:
                save_path = f"{name}/{image['id']}.jpg"
                if counter == max_images:
                    break
                if os.path.isfile(save_path):
                    downloaded_images.append(save_path)
                    counter += 1
                    continue
                try:
                    response = requests.get(image["url"], headers=headers, timeout=timeout)
                    # Check if the request was successful (status code 200)
                    if response.status_code == 200:
                        # Open the file in binary write mode and write the content
                        binary_content = str(response.content)
                        if "DOCTYPE" in binary_content or "html" in binary_content or "doctype" in binary_content or "HTML" in binary_content:
                            continue
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        print("\nImage downloaded successfully!", save_path, "\n")
                        downloaded_images.append(save_path)
                        counter += 1
                    else:
                        print("Failed to download image. Status code:", response.status_code)
                except requests.Timeout:
                    print("Request timed out after", timeout, "seconds.")
                except Exception as e:
                    print("An error occurred:", str(e)[0:50])
    # save the downloaded images to a json file
    json.dump(downloaded_images, open(downloaded_images_json, "w"), indent=4)
    os.chdir("..")
    
def show_image(image, x1, y1, x2, y2, title="Bounding Box", bbox=True):
    if bbox:
        image = cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.show()

def get_eye_cascade():
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    return eye_cascade

def salient_features(image_path, output_path, face):
    # Load the image
    image = cv.imread(image_path)
    if image is None:
        print("Image not found")
        return

    if not os.path.isfile(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    faces = face

    # Apply aggressive feature obfuscation while maintaining recognizability for humans
    for (x1, y1, x2, y2) in faces:
        # Extract the face region
        face_region = image[y1:y2, x1:x2]
        # Detect the eyes
        eye_cascade = get_eye_cascade()
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_region = face_region.astype(np.float32)
        # Darken the eyes
        for (ex, ey, ew, eh) in eyes:
            for x in range(ex, ex + ew):
                for y in range(ey, ey + eh):
                    face_region[y, x] *= 0.5
        face_region = np.clip(face_region, 0, 255).astype(np.uint8)
        image[y1:y2, x1:x2] = face_region

    # Save the modified image
    cv.imwrite(output_path, image)

def bbox(image_path, output_path, face):
    # Load the image
    image = cv.imread(image_path)
    if image is None:
        print("Image not found")
        return
    
    if not os.path.isfile(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Apply boundary box obfuscation with modified appearance
    for (x1, y1, x2, y2) in face:
        # Draw a thicker white rectangle around the face region
        cv.rectangle(image, (x1-10, y1-10), (x2+10, y2+10), (255, 255, 255), 2)  # White rectangle with thickness 2
        image = image.astype(np.float32)
        for x in range(x1-10, x2+10):
            for y in range(y1-10, y2+10):
                image[y, x] *= 0.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        # Draw a thinner black rectangle inside the white one
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)  # Black rectangle with thickness 1

    # Save the modified image
    cv.imwrite(output_path, image)

def choose_random(count, choose_new):
    # get 200 random people
    # if chosen_people.json exists, load the chosen people from the json file
    if os.path.isfile("chosen_people.json") and not choose_new:
        chosen_people = json.load(open("chosen_people.json", "r"))
    else:
        people = list(images.keys())
        chosen_people = random.sample(people, count)
        json.dump(chosen_people, open("chosen_people.json", "w"), indent=4)

    # read downloaded images
    downloaded_images = json.load(open(downloaded_images_json, "r"))

    # get the first image of each person from the downloaded images
    chosen_images = {}
    for person in chosen_people:
        for image in downloaded_images:
            if person in image:
                chosen_images[person] = downloaded_images[image][0:2]
                break

    return chosen_images

def apply_perturbation(chosen_images):
    # read images.json
    images = json.load(open(json_file, "r"))

    for i in chosen_images:
        for j in chosen_images[i]:
            try:
                image_data = images[i]
                for image in image_data:
                    if image["image_id"] == j:
                        face = image["face_location"]
                        # show_image(img, face["x1"], face["y1"], face["x2"], face["y2"], i, True)
                        salient_features(f"org_images/{i}/{j}.jpg", f"salient/{i}_{j}.jpg", [(int(face["x1"]), int(face["y1"]), int(face["x2"]), int(face["y2"]))])
                        bbox(f"org_images/{i}/{j}.jpg", f"bbox/{i}_{j}.jpg", [(int(face["x1"]), int(face["y1"]), int(face["x2"]), int(face["y2"]))])
            except Exception as e:
                print("An error occurred:", str(e))
                break

def get_all_images(chosen_images, images):
    all_images = {}

    for i in chosen_images:
        for j in chosen_images[i]:
            image_data = images[i]
            for image in image_data:
                if image["image_id"] == j:
                    face = image["face_location"]
                    break
            # get the original image
            img = cv.imread(f"org_images/{i}/{j}.jpg")
            # get the salient image
            salient = cv.imread(f"salient/{i}_{j}.jpg")
            # get the bbox image
            bbox = cv.imread(f"bbox/{i}_{j}.jpg")
            # show the images
            # show_image(img, 0, 0, 0, 0, f"{i} Original")
            # show_image(salient, 0, 0, 0, 0, f"{i} Salient")
            # show_image(bbox, 0, 0, 0, 0, f"{i} Bbox")
            if i not in all_images:
                all_images[i] = {}
            all_images[i][j] = {
                "original": img,
                "salient": salient,
                "bbox": bbox,
                "face": [(int(face["x1"]), int(face["y1"]), int(face["x2"]), int(face["y2"]))]
            }
    return all_images

def plot_single_celeb(all_images):
    for key, value in all_images.items():
        originals = []
        salients = []
        bboxes = []
        for k, v in value.items():
            originals.append(v["original"])
            salients.append(v["salient"])
            bboxes.append(v["bbox"])
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(cv.cvtColor(originals[0], cv.COLOR_BGR2RGB))
        ax[0].set_title(f"{key} Original")
        ax[0].axis("off")
        ax[1].imshow(cv.cvtColor(salients[0], cv.COLOR_BGR2RGB))
        ax[1].set_title(f"{key} Salient")
        ax[1].axis("off")
        ax[2].imshow(cv.cvtColor(bboxes[0], cv.COLOR_BGR2RGB))
        ax[2].set_title(f"{key} Bbox")
        ax[2].axis("off")
        plt.show()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(cv.cvtColor(originals[1], cv.COLOR_BGR2RGB))
        ax[0].set_title(f"{key} Original")
        ax[0].axis("off")
        ax[1].imshow(cv.cvtColor(salients[1], cv.COLOR_BGR2RGB))
        ax[1].set_title(f"{key} Salient")
        ax[1].axis("off")
        ax[2].imshow(cv.cvtColor(bboxes[1], cv.COLOR_BGR2RGB))
        ax[2].set_title(f"{key} Bbox")
        ax[2].axis("off")
        plt.show()
        break

def calculate_metrics_positive(all_images, correct_matches_second=0, total_matches_second=0, correct_matches_salient=0, total_matches_salient=0, correct_matches_bbox=0, total_matches_bbox=0, correct_matches_salient_bbox=0, total_matches_salient_bbox=0):
    for key, value in all_images.items():
        originals = []
        salients = []
        bboxes = []
        face_locations = []
        for k, v in value.items():
            originals.append(v["original"])
            salients.append(v["salient"])
            bboxes.append(v["bbox"])
            face_locations.append(v["face"])

        reference_image = cv.cvtColor(originals[0], cv.COLOR_BGR2RGB)
        reference_face_encoding = face_recognition.face_encodings(reference_image, known_face_locations=face_locations[0])[0]

        test_image_second = cv.cvtColor(originals[1], cv.COLOR_BGR2RGB)
        test_face_encoding_second = face_recognition.face_encodings(test_image_second, known_face_locations=face_locations[1])[0]

        test_image_salient = cv.cvtColor(salients[1], cv.COLOR_BGR2RGB)
        test_face_encoding_salient = face_recognition.face_encodings(test_image_salient, known_face_locations=face_locations[1])[0]

        test_image_bbox = cv.cvtColor(bboxes[1], cv.COLOR_BGR2RGB)
        test_face_encoding_bbox = face_recognition.face_encodings(test_image_bbox, known_face_locations=face_locations[1])[0]


        match = face_recognition.compare_faces([reference_face_encoding], test_face_encoding_second)
        if match[0]:
            correct_matches_second += 1
        total_matches_second += 1

        match = face_recognition.compare_faces([reference_face_encoding], test_face_encoding_salient)
        if match[0]:
            correct_matches_salient += 1
        total_matches_salient += 1

        match = face_recognition.compare_faces([reference_face_encoding], test_face_encoding_bbox)
        if match[0]:
            correct_matches_bbox += 1
        total_matches_bbox += 1

        match = face_recognition.compare_faces([test_face_encoding_salient], test_face_encoding_bbox)
        if match[0]:
            correct_matches_salient_bbox += 1
        total_matches_salient_bbox += 1

    tp_second = correct_matches_second
    fp_second = total_matches_second - correct_matches_second

    tp_salient = correct_matches_salient
    fp_salient = total_matches_salient - correct_matches_salient

    tp_bbox = correct_matches_bbox
    fp_bbox = total_matches_bbox - correct_matches_bbox

    tp_salient_bbox = correct_matches_salient_bbox
    fp_salient_bbox = total_matches_salient_bbox - correct_matches_salient_bbox

    return tp_second, fp_second, tp_salient, fp_salient, tp_bbox, fp_bbox, tp_salient_bbox, fp_salient_bbox

def calculate_metrics_negative(all_images, negative_matches_second=0, total_negative_matches_second=0, negative_matches_salient=0, total_negative_matches_salient=0, negative_matches_bbox=0, total_negative_matches_bbox=0, negative_matches_salient_bbox=0, total_negative_matches_salient_bbox=0):
    for key, value in all_images.items():
        first = True
        for k, v in value.items():
            if first:
                originals_1.append(v["original"])
                salients_1.append(v["salient"])
                bboxes_1.append(v["bbox"])
                face_locations_1.append(v["face"])
                first = False
            else:
                originals_2.append(v["original"])
                salients_2.append(v["salient"])
                bboxes_2.append(v["bbox"])
                face_locations_2.append(v["face"])

    choosing_list_check = list(all_images.keys())
    choosing_list = list(all_images.keys())

    for i in range(200):
        if len(choosing_list) < 2:
            print("Not enough people to compare")
            break
        random_person_1 = random.choice(choosing_list)
        index_person_1 = choosing_list_check.index(random_person_1)
        choosing_list.remove(random_person_1)

        random_person_2 = random.choice(choosing_list)
        index_person_2 = choosing_list_check.index(random_person_2)
        choosing_list.remove(random_person_2)

        reference_image = cv.cvtColor(originals_1[index_person_1], cv.COLOR_BGR2RGB)
        reference_face_encoding = face_recognition.face_encodings(reference_image, known_face_locations=face_locations_1[index_person_1])[0]

        test_image_second = cv.cvtColor(originals_2[index_person_2], cv.COLOR_BGR2RGB)
        test_face_encoding_second = face_recognition.face_encodings(test_image_second, known_face_locations=face_locations_2[index_person_2])[0]

        test_image_salient = cv.cvtColor(salients_2[index_person_2], cv.COLOR_BGR2RGB)
        test_face_encoding_salient = face_recognition.face_encodings(test_image_salient, known_face_locations=face_locations_2[index_person_2])[0]

        test_image_bbox = cv.cvtColor(bboxes_2[index_person_2], cv.COLOR_BGR2RGB)
        test_face_encoding_bbox = face_recognition.face_encodings(test_image_bbox, known_face_locations=face_locations_2[index_person_2])[0]


        match = face_recognition.compare_faces([reference_face_encoding], test_face_encoding_second)
        if match[0]:
            negative_matches_second += 1
        total_negative_matches_second += 1

        match = face_recognition.compare_faces([reference_face_encoding], test_face_encoding_salient)
        if match[0]:
            negative_matches_salient += 1
        total_negative_matches_salient += 1

        match = face_recognition.compare_faces([reference_face_encoding], test_face_encoding_bbox)
        if match[0]:
            negative_matches_bbox += 1
        total_negative_matches_bbox += 1

        match = face_recognition.compare_faces([test_face_encoding_salient], test_face_encoding_bbox)
        if match[0]:
            negative_matches_salient_bbox += 1
        total_negative_matches_salient_bbox += 1

        tn_second = negative_matches_second
        fn_second = total_negative_matches_second - negative_matches_second

        tn_salient = negative_matches_salient
        fn_salient = total_negative_matches_salient - negative_matches_salient

        tn_bbox = negative_matches_bbox
        fn_bbox = total_negative_matches_bbox - negative_matches_bbox

        tn_salient_bbox = negative_matches_salient_bbox
        fn_salient_bbox = total_negative_matches_salient_bbox - negative_matches_salient_bbox

    return tn_second, fn_second, tn_salient, fn_salient, tn_bbox, fn_bbox, tn_salient_bbox, fn_salient_bbox

def calculate_all_metrics(type, tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print(type, " Accuracy:", accuracy)
    print(type, " Precision:", precision)
    print(type, " Recall:", recall)
    print(type, " F1 Score:", f1_score)

    return accuracy, precision, recall, f1_score


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    # parser.add_argument("get_metrics", help="Save metrics to a JSON file.", type=bool)

    # Optional arguments
    parser.add_argument("-m", "--metrics", help="Save metrics to a JSON file.", type=bool, default=False)
    parser.add_argument("-ni", "--num_images", help="Number of images per person to download.", type=int, default=3)
    parser.add_argument("-c", "--count", help="Number of people to choose.", type=int, default=200)
    parser.add_argument("-n", "--new", help="Choose new people.", type=bool, default=False)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    os.chdir("salient_feature_obfuscation")

    args = parseArguments()

    json_file = "images.json"

    downloaded_images_json = "downloaded_images.json"

    originals_1 = []
    salients_1 = []
    bboxes_1 = []
    face_locations_1 = []

    originals_2 = []
    salients_2 = []
    bboxes_2 = []
    face_locations_2 = []

    images = organize_metadata()

    num_images = args.num_images

    download_image(images, num_images)

    count = args.count
    choose_new = args.new

    chosen_images = choose_random()

    apply_perturbation(chosen_images)

    all_images = get_all_images(chosen_images, images)

    # plot_single_celeb(all_images)

    tp_second, fp_second, tp_salient, fp_salient, tp_bbox, fp_bbox, tp_salient_bbox, fp_salient_bbox = calculate_metrics_positive(all_images)

    tn_second, fn_second, tn_salient, fn_salient, tn_bbox, fn_bbox, tn_salient_bbox, fn_salient_bbox = calculate_metrics_negative(all_images)

    accuracy_second, precision_second, recall_second, f1_score_second = calculate_all_metrics("Original Images Match",tp_second, fp_second, tn_second, fn_second)
    accuracy_salient, precision_salient, recall_salient, f1_score_salient = calculate_all_metrics("Salient Feature Obfuscation Match",tp_salient, fp_salient, tn_salient, fn_salient)
    accuracy_bbox, precision_bbox, recall_bbox, f1_score_bbox = calculate_all_metrics("Whole Face Obfuscation Match",tp_bbox, fp_bbox, tn_bbox, fn_bbox)
    accuracy_salient_bbox, precision_salient_bbox, recall_salient_bbox, f1_score_salient_bbox = calculate_all_metrics("Salient Feature vs Whole Face Obfuscation Match",tp_salient_bbox, fp_salient_bbox, tn_salient_bbox, fn_salient_bbox)

    if args.metrics:
        metrics = {}
        metrics_second = {
            "accuracy": accuracy_second,
            "precision": precision_second,
            "recall": recall_second,
            "f1_score": f1_score_second
        }
        metrics_salient = {
            "accuracy": accuracy_salient,
            "precision": precision_salient,
            "recall": recall_salient,
            "f1_score": f1_score_salient
        }
        metrics_bbox = {
            "accuracy": accuracy_bbox,
            "precision": precision_bbox,
            "recall": recall_bbox,
            "f1_score": f1_score_bbox
        }
        metrics_salient_bbox = {
            "accuracy": accuracy_salient_bbox,
            "precision": precision_salient_bbox,
            "recall": recall_salient_bbox,
            "f1_score": f1_score_salient_bbox
        }
        metrics["Originals"] = metrics_second
        metrics["Salient"] = metrics_salient
        metrics["Face"] = metrics_bbox
        metrics["Salient_vs_Face"] = metrics_salient_bbox
