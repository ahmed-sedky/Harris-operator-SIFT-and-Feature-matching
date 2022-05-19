import numpy as np
import cv2
import algorithms.sift as SIFT


def matching(descriptor1, descriptor2, match_calculator):
    keypoints1 = descriptor1.shape[0]
    keypoints2 = descriptor2.shape[0]
    matches = []

    for kp1 in range(keypoints1):
        distance = -np.inf
        y_index = -1
        for kp2 in range(keypoints2):
            value = match_calculator(descriptor1[kp1], descriptor2[kp2])
            if value > distance:
                distance = value
                y_index = kp2

        match = cv2.DMatch()
        match.queryIdx = kp1
        match.trainIdx = y_index
        match.distance = distance
        matches.append(match)
    matches = sorted(matches, key=lambda x: x.distance, reverse=True)
    return matches


def calculate_ncc(descriptor1, descriptor2):
    out1_normalized = (descriptor1 - np.mean(descriptor1)) / (np.std(descriptor1))
    out2_normalized = (descriptor2 - np.mean(descriptor2)) / (np.std(descriptor2))

    correlation_vector = np.multiply(out1_normalized, out2_normalized)

    correlation = float(np.mean(correlation_vector))

    return correlation


def calculate_ssd(descriptor1, descriptor2):
    ssd = 0
    for m in range(len(descriptor1)):
        ssd += (descriptor1[m] - descriptor2[m]) ** 2

    ssd = -(np.sqrt(ssd))
    return ssd


def SSD(image_1, image_2):
    image_1_copy = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2_copy = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    keypoints_1, descriptor_1 = SIFT.SIFT(image_1_copy)
    keypoints_2, descriptor_2 = SIFT.SIFT(image_2_copy)

    matches_ssd = matching(descriptor_1, descriptor_2, calculate_ssd)
    matched_image_ssd = cv2.drawMatches(
        image_1_copy,
        keypoints_1,
        image_2_copy,
        keypoints_2,
        matches_ssd[:30],
        image_2_copy,
        flags=2,
    )

    return matched_image_ssd


def NCC(image_1, image_2):
    image_1_copy = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2_copy = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptor_1 = SIFT.SIFT(image_1_copy)
    keypoints_2, descriptor_2 = SIFT.SIFT(image_2_copy)

    matches_ncc = matching(descriptor_1, descriptor_2, calculate_ncc)
    matched_image_ncc = cv2.drawMatches(
        image_1_copy,
        keypoints_1,
        image_2_copy,
        keypoints_2,
        matches_ncc[:30],
        image_2_copy,
        flags=2,
    )

    return matched_image_ncc
