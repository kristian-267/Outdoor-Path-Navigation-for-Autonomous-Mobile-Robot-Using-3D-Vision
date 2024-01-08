import numpy as np

class Voxelize():
    def __init__(self,
                 voxel_size=0.05,
                 mode='train',
                 keys=("coord", "color"),
                 return_discrete_coord=True):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(int)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == 'train':  # train mode
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict, idx_unique
        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr