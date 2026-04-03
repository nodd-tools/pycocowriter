import random
import copy

def split_coco_by_image_ids(coco_dict: dict, *split_img_ids: set[int]) -> tuple[dict, ...]:
    """
    Splits a COCO dictionary into N new dictionaries based on pre-computed image ID sets.
    
    This abstracts the tedious dictionary reconstruction logic. It preserves the
    metadata (info, licenses, categories) in all outputs, and strictly routes 
    the images and annotations into their respective splits.

    Parameters
    ----------
    coco_dict : dict
        The source COCO dictionary.
    *split_img_ids : set[int]
        A variable number of sets containing image IDs for each desired split.

    Returns
    -------
    tuple[dict, ...]
        A tuple of COCO dictionaries, mapped 1-to-1 with the provided image ID sets.

    Examples
    --------
    >>> coco = {
    ...     "images": [{"id": 1}, {"id": 2}, {"id": 3}],
    ...     "annotations": [{"id": 10, "image_id": 1}, {"id": 20, "image_id": 2}]
    ... }
    >>> s1, s2, s3 = split_coco_by_image_ids(coco, {1}, {2}, {3})
    >>> s1["images"]
    [{'id': 1}]
    >>> s2["annotations"]
    [{'id': 20, 'image_id': 2}]
    >>> s3["images"]
    [{'id': 3}]
    """
    def _build_split(target_ids: set[int]) -> dict:
        return {
            "info": copy.deepcopy(coco_dict.get("info", {})),
            "licenses": copy.deepcopy(coco_dict.get("licenses", [])),
            "categories": copy.deepcopy(coco_dict.get("categories", [])),
            "images": [img for img in coco_dict.get("images", []) if img["id"] in target_ids],
            "annotations": [ann for ann in coco_dict.get("annotations", []) if ann["image_id"] in target_ids]
        }
        
    return tuple(_build_split(ids) for ids in split_img_ids)


def naive_random_split(
    coco_dict: dict, 
    split_ratios: tuple[float, ...] | list[float] = (0.85, 0.15), 
    seed: int = 42
) -> tuple[dict, ...]:
    """
    Performs a completely naive image-level random N-way split.
    
    This strategy ignores annotations and filenames entirely. It simply shuffles 
    the images and splits them. It is fast but highly susceptible to data leakage 
    in video datasets and may result in the loss of rare taxonomic classes.

    Parameters
    ----------
    coco_dict : dict
        The source COCO dictionary.
    split_ratios : tuple[float, ...] | list[float], optional
        The proportional sizes of the target splits (e.g., [0.7, 0.15, 0.15]). 
        These values will be automatically normalized to sum to 1.0. 
        Default is (0.85, 0.15).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    tuple[dict, ...]
        A tuple of COCO dictionaries corresponding to the requested splits.

    Examples
    --------
    >>> coco = {"images": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]}
    >>> s1, s2 = naive_random_split(coco, split_ratios=(0.5, 0.5), seed=42)
    >>> len(s1["images"])
    2
    >>> len(s2["images"])
    2
    """
    rng = random.Random(seed)
    
    img_ids = [img['id'] for img in coco_dict.get('images', [])]
    rng.shuffle(img_ids)
    
    total_ratio = sum(split_ratios)
    ratios = [r / total_ratio for r in split_ratios]
    
    split_img_ids = []
    prev_idx = 0
    cum_ratio = 0.0
    
    for r in ratios:
        cum_ratio += r
        idx = round(len(img_ids) * cum_ratio)
        split_img_ids.append(set(img_ids[prev_idx:idx]))
        prev_idx = idx
        
    return split_coco_by_image_ids(coco_dict, *split_img_ids)


# ==========================================
# Rarity Stratification Engine
# ==========================================

class RarityStratifier:
    """
    Class-based state manager for iterative, rarity-based multi-label stratification.
    
    This engine solves the "Passenger Problem" in multi-label images by sorting 
    classes ascending by rarity, tracking running deficits, and water-falling 
    images to target split proportions.
    
    Parameters
    ----------
    coco_dict : dict
        The source COCO dictionary containing images and annotations.
    split_ratios : list[float] | tuple[float, ...]
        The proportional sizes of the target splits.
    sort_by_filename : bool
        If True, unassigned image pools are sorted by filename before being sliced. 
        This mitigates data leakage in sequential video frames.
    seed : int
        Random seed for shuffling unassigned pools and waterfall targets.
    """
    def __init__(
        self, 
        coco_dict: dict, 
        split_ratios: list[float] | tuple[float, ...], 
        sort_by_filename: bool, 
        seed: int
    ) -> None:
        self.coco_dict = coco_dict
        self.sort_by_filename = sort_by_filename
        self.rng = random.Random(seed)
        
        total_ratio = sum(split_ratios)
        self.ratios = [r / total_ratio for r in split_ratios]
        self.num_splits = len(self.ratios)
        
        # Build initial bipartite graph lookups
        self.img_id_to_filename, self.img_to_cats, self.cat_to_imgs = self.build_category_lookups()
        
        # Initialize running tracking states
        self.split_img_ids = [set() for _ in range(self.num_splits)]
        self.split_counts = [{c: 0 for c in self.cat_to_imgs} for _ in range(self.num_splits)]
        
        # Flat O(1) lookup set to replace O(N x S) iteration during pool construction
        self.assigned_img_ids = set()

    def build_category_lookups(self) -> tuple[dict, dict, dict]:
        """
        Builds graph mappings between images and categories.

        Returns
        -------
        tuple[dict[int, str], dict[int, set[int]], dict[int, set[int]]]
            - img_id_to_filename: Maps image IDs to their filenames.
            - img_to_cats: Maps an image ID to a set of category IDs it contains.
            - cat_to_imgs: Maps a category ID to a set of image IDs containing it.
        """
        img_id_to_filename = {img['id']: img.get('file_name', '') for img in self.coco_dict.get('images', [])}
        img_to_cats = {img['id']: set() for img in self.coco_dict.get('images', [])}
        
        for ann in self.coco_dict.get('annotations', []):
            if ann['image_id'] in img_to_cats:
                img_to_cats[ann['image_id']].add(ann['category_id'])
                
        cat_to_imgs = {}
        for img_id, cats in img_to_cats.items():
            for cat in cats:
                if cat not in cat_to_imgs:
                    cat_to_imgs[cat] = set()
                cat_to_imgs[cat].add(img_id)
                
        return img_id_to_filename, img_to_cats, cat_to_imgs

    def calculate_cumulative_targets(self, total_items: int) -> list[int]:
        """
        Calculates exact allocation targets using Cumulative Rounding to avoid 
        the Rounding Paradox across arbitrary N-way splits.

        Parameters
        ----------
        total_items : int
            The total population size to compute target splits against.

        Returns
        -------
        list[int]
            A list of integer targets corresponding to the ideal size of each split.
        """
        targets = []
        cum_ratio = 0.0
        prev_target = 0
        for r in self.ratios:
            cum_ratio += r
            cum_target = round(total_items * cum_ratio)
            targets.append(cum_target - prev_target)
            prev_target = cum_target
        return targets

    def waterfall_assign(self, unassigned: list[int], deficits: list[int]) -> None:
        """
        Randomizes the split order and waterfalls unassigned images to fulfill 
        deficits, updating internal tracking states in-place.

        Parameters
        ----------
        unassigned : list[int]
            A pool of image IDs available to be allocated.
        deficits : list[int]
            A list indicating how many more images each split currently needs.
        """
        split_indices = list(range(self.num_splits))
        self.rng.shuffle(split_indices)
        
        for i in split_indices:
            needed = max(0, min(deficits[i], len(unassigned)))
            to_split = unassigned[:needed]
            unassigned = unassigned[needed:]
            
            # Update state maps
            self.split_img_ids[i].update(to_split)
            self.assigned_img_ids.update(to_split)
            
            # Update running counts for all passenger categories in the allocated images
            for img_id in to_split:
                for c in self.img_to_cats[img_id]:
                    self.split_counts[i][c] += 1

    def allocate_pool(self, pool: list[int], population_size: int, current_counts: list[int]) -> None:
        """
        Orchestrates target calculation, pool sorting, and execution of the 
        waterfall assignment.

        Parameters
        ----------
        pool : list[int]
            The available pool of unassigned image IDs.
        population_size : int
            The total original size of the population this pool was drawn from.
        current_counts : list[int]
            The current number of allocated images/annotations for each split.
        """
        if not pool:
            return
            
        targets = self.calculate_cumulative_targets(population_size)
        deficits = [targets[i] - current_counts[i] for i in range(self.num_splits)]
        
        if self.sort_by_filename:
            pool.sort(key=lambda x: self.img_id_to_filename.get(x, ''))
        else:
            self.rng.shuffle(pool)
            
        self.waterfall_assign(pool, deficits)

    def split(self) -> tuple[dict, ...]:
        """
        Executes the full iterative rarity stratification pipeline.

        Returns
        -------
        tuple[dict, ...]
            A tuple of COCO dictionaries corresponding to the requested splits.
        """
        cat_order = sorted(self.cat_to_imgs.keys(), key=lambda c: len(self.cat_to_imgs[c]))
        
        # Phase 1: Iterative Rarity Assignment
        for cat in cat_order:
            unassigned = [
                img_id for img_id in self.cat_to_imgs[cat]
                if img_id not in self.assigned_img_ids
            ]
            
            current_counts = [self.split_counts[i][cat] for i in range(self.num_splits)]
            
            self.allocate_pool(
                pool=unassigned,
                population_size=len(self.cat_to_imgs[cat]),
                current_counts=current_counts
            )
                    
        # Phase 2: Catch-all for empty images or leftovers
        all_imgs = set(self.img_id_to_filename.keys())
        leftovers = list(all_imgs - self.assigned_img_ids)
        
        current_counts = [len(self.split_img_ids[i]) for i in range(self.num_splits)]
        
        self.allocate_pool(
            pool=leftovers,
            population_size=len(all_imgs),
            current_counts=current_counts
        )
                
        return split_coco_by_image_ids(self.coco_dict, *self.split_img_ids)


def rarity_stratified_split(
    coco_dict: dict, 
    split_ratios: tuple[float, ...] | list[float] = (0.85, 0.15), 
    sort_by_filename: bool = False, 
    seed: int = 42
) -> tuple[dict, ...]:
    """
    Performs an iterative, rarity-based multi-label N-way stratified split.
    
    This wrapper leverages the `RarityStratifier` to solve the "Passenger Problem" 
    common in multi-label images. It ensures rare taxonomic classes are proportionately 
    split before common classes swallow them up as background noise.

    Parameters
    ----------
    coco_dict : dict
        The source COCO dictionary.
    split_ratios : tuple[float, ...] | list[float], optional
        The proportional sizes of the target splits (e.g., [0.7, 0.15, 0.15]). 
        These values will be automatically normalized to sum to 1.0. 
        Default is (0.85, 0.15).
    sort_by_filename : bool, optional
        If True, unassigned image pools are sorted by filename before being sliced. 
        This mitigates data leakage in sequential video frames. If False, the pool 
        is randomly shuffled. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    tuple[dict, ...]
        A tuple of COCO dictionaries corresponding to the requested splits.

    Examples
    --------
    >>> coco = {
    ...     "images": [
    ...         {"id": 1, "file_name": "a.jpg"},
    ...         {"id": 2, "file_name": "b.jpg"},
    ...         {"id": 3, "file_name": "c.jpg"}
    ...     ],
    ...     "annotations": [
    ...         {"id": 1, "image_id": 1, "category_id": 10}, # Rare
    ...         {"id": 2, "image_id": 2, "category_id": 20}, # Common
    ...         {"id": 3, "image_id": 3, "category_id": 20}  # Common
    ...     ]
    ... }
    >>> t, v = rarity_stratified_split(coco, split_ratios=[0.6, 0.4], sort_by_filename=True, seed=42)
    >>> len(t["images"])
    2
    >>> len(v["images"])
    1
    """
    stratifier = RarityStratifier(
        coco_dict=coco_dict,
        split_ratios=split_ratios,
        sort_by_filename=sort_by_filename,
        seed=seed
    )
    return stratifier.split()
