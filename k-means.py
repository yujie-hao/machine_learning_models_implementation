import random


class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()


def get_k_means(user_feature_map, num_features_per_user, k):
    # Don't change the following two lines of code.
    random.seed(42)

    # step-1: Gets the initial samples, to be used as centroids.
    initial_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)
    new_centroids = [Centroid(user_feature_map[user]) for user in initial_centroid_users]

    for i in range(10):
        # step-2: assign all samples to their nearest centroids
        assign_to_centroids(user_feature_map, new_centroids)
        # step-3: find new centroids based on average feature value
        new_centroids = find_new_centroid(user_feature_map, num_features_per_user, new_centroids)
    return [c.location for c in new_centroids]


def find_new_centroid(user_feature_map, num_features_per_user, centroids):
    new_centroids = []
    for c in centroids:
        tempCentroidSums = [0] * num_features_per_user
        for uid in c.closest_users:
            for i in range(len(user_feature_map[uid])):
                tempCentroidSums[i] += user_feature_map[uid][i]
        tempCentroidLoc = [x / len(c.closest_users) for x in tempCentroidSums]
        tempCentroid = Centroid(tempCentroidLoc)
        new_centroids.append(tempCentroid)
    return new_centroids


def assign_to_centroids(user_feature_map, centroids):
    for uid, feature in user_feature_map.items():
        minManhattanDistance = float("inf")
        centroidIndex = -1
        for i in range(len(centroids)):
            tempDistance = get_manhattan_distance(centroids[i], feature)
            if tempDistance < minManhattanDistance:
                centroidIndex = i
                minManhattanDistance = tempDistance
        centroids[centroidIndex].closest_users.add(uid)


def get_manhattan_distance(centroid, user_feature):
    manhattan_distance = 0
    for i in range(len(user_feature)):
        manhattan_distance += abs(centroid.location[i] - user_feature[i])
    return manhattan_distance
