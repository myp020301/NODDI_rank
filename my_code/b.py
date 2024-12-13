import numpy as np
import nibabel as nib
import os


def angle_between_vectors(vec1, vec2):
    """计算两个向量之间的夹角（以弧度为单位）。"""
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    return angle

def cosine_distance(vec1, vec2):
    """计算两个向量的余弦距离，距离定义为两向量夹角的余弦值。
       如果夹角大于90度，则将其中一个向量取反方向。"""
    angle = angle_between_vectors(vec1, vec2)
    if angle > np.pi / 2:
        # 如果夹角大于90度，取其中一个向量的反方向
        angle = angle_between_vectors(-vec1, vec2)
    return 1 - np.cos(angle)

def find_closest_vectors(target_vectors, candidate_vectors, candidate_indices):
    """寻找与目标向量夹角最小的向量。"""
    closest_indices = []
    for target in target_vectors:
        distances = [cosine_distance(target, candidate_vectors[i]) for i in candidate_indices]
        sorted_indices = np.argsort(distances)
        closest_indices.append(candidate_indices[sorted_indices[0]])
    return list(closest_indices)


def process_subject_data(subject_folder, bval_target, selected_bvecs_path, num_vectors):
    data_path = os.path.join(subject_folder, "data_normalize_dwi.nii.gz")
    bval_path = os.path.join(subject_folder, "data_normalize_b.txt")
    bvec_path = os.path.join(subject_folder, "data_normalize_grad.txt")

    data_img = nib.load(data_path)
    data = data_img.get_fdata()
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)

    target_indices = np.where(bvals == bval_target)[0]

    target_bvecs = bvecs[target_indices]
    
    selected_bvecs = np.loadtxt(selected_bvecs_path)

    closest_indices = find_closest_vectors(selected_bvecs, bvecs, target_indices)
    combined_data = data[..., closest_indices]
    
    return combined_data, data_img


subject_list_path = "/data2/mayupeng/HCP_25_dataset_ori/subjectlist.txt"
with open(subject_list_path, "r") as file:
    subjects = file.read().splitlines()


for subject in subjects:
    subject_folder = os.path.join("/data2/mayupeng/HCP_25_dataset_ori", subject)
    combined_images = []
    reference_img = None
    combined_1000, img_1000 = process_subject_data(
        subject_folder, 1000, "/data2/mayupeng/HCP_25_dataset_ori/100206/b1000_30.txt", 30
    )
    combined_2000, img_2000 = process_subject_data(
        subject_folder, 2000, "/data2/mayupeng/HCP_25_dataset_ori/100206/b2000_30.txt", 30
    )

    if combined_1000.shape[3] == 1:
        combined_1000 = np.squeeze(combined_1000, axis=3)
    if combined_2000.shape[3] == 1:
        combined_2000 = np.squeeze(combined_2000, axis=3)
        
    combined_images.extend([combined_1000, combined_2000])
    reference_img = img_1000
    final_combined_image = np.concatenate(combined_images, axis=3)
    final_img = nib.Nifti1Image(final_combined_image, reference_img.affine)
    nib.save(final_img, os.path.join(subject_folder, "data_fi.nii.gz"))
    print(f"处理完成：受试者 {subject} 的数据已合并。")


print("所有受试者的处理和图像合并已完成。")