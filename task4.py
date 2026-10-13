# %%
# Imports
# ================================================================
# Don't update these imports.
import os
import cv2
import getpass
import numpy as np
import matplotlib.pyplot as plt
from task1 import *
from task2 import *
from task3 import *
# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)


def with_unit_scale(keypoints: np.ndarray) -> np.ndarray:
    """Give scale-less keypoints a sigma of 1, so (N, 2) and (N, 3) both work.

    Harris corners are (y, x): a corner has no size of its own. Padding them
    to (y, x, 1) lets the same matching code take either detector -- at the
    cost of the scale ratio then carrying no information at all.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    kp = kp.reshape(-1, kp.shape[-1]) if kp.size else kp.reshape(0, 3)
    if kp.shape[1] >= 3:
        return kp
    return np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=1)


def box_from_centre(cx: float, cy: float, scale: float,
                    template_shape: tuple) -> tuple:
    """A box the shape of the template, scaled and centred on (cx, cy)."""
    th, tw = template_shape
    w = max(1, int(round(tw * scale)))
    h = max(1, int(round(th * scale)))
    return (int(np.floor(cx - w / 2.0)), int(np.floor(cy - h / 2.0)), w, h)


def draw_matches(template: np.ndarray, scene_rgb: np.ndarray,
                 kp_t: np.ndarray, kp_s: np.ndarray,
                 matches: np.ndarray, votes: np.ndarray, path: str) -> None:
    """Template and scene side by side, with a line for every match."""
    zoom = 2
    left = np.repeat(np.repeat(np.dstack([template] * 3), zoom, 0), zoom, 1)
    height = max(left.shape[0], scene_rgb.shape[0])
    canvas = np.ones((height, left.shape[1] + scene_rgb.shape[1], 3))
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:scene_rgb.shape[0], left.shape[1]:] = scene_rgb

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas)
    

    for (a, b), voted in zip(matches, votes):
        plt.plot(
            [kp_t[a, 1] * zoom / 2, kp_s[b, 1] + left.shape[1]],
            [kp_t[a, 0] * zoom / 2, kp_s[b, 0]],
            color="lime" if voted else "red",
            linewidth=0.7,
            alpha=0.85
        )


    plt.title("green = agreed with the consensus, red = outvoted")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, metadata={"Author": getpass.getuser()})
    plt.close()


# ================================================================
# %%
# Task 4
# ======


def descriptor_distances(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """Distance from every descriptor in A to every descriptor in B.

    Parameters
    ----------
    desc_a : np.ndarray
        Shape (Na, D)
    desc_b : np.ndarray
        Shape (Nb, D)

    Returns
    -------
    np.ndarray
        Shape (Na, Nb) of Euclidean distances
    """
    a=np.asarray(desc_a,float); b=np.asarray(desc_b,float)
    if len(a)==0 or len(b)==0: return np.empty((len(a),len(b)))
    aa=np.sum(a*a,axis=1,keepdims=True); bb=np.sum(b*b,axis=1,keepdims=True).T
    return np.sqrt(np.maximum(aa+bb-2*a@b.T,0))
    # return np.zeros((len(desc_a), len(desc_b)))  # comment this line and write your code for the function


def match_ratio_test(desc_scene: np.ndarray, desc_template: np.ndarray,
                     ratio: float = 0.8) -> np.ndarray:
    """Ask every SCENE keypoint which template part it looks like.

    Parameters
    ----------
    desc_scene : np.ndarray
        Shape (Ns, D), the queries
    desc_template : np.ndarray
        Shape (Nt, D), the small database
    ratio : float
        Lowe's threshold

    Returns
    -------
    np.ndarray
        Shape (M, 2) of (template index, scene index), best match first.
        (0, 2) if nothing survives. Note the column order: template first,
        which is what predict_centres expects.
    """
    if len(desc_scene) == 0 or len(desc_template) < 2:
        return np.zeros((0, 2), dtype=int)

    d = descriptor_distances(desc_scene, desc_template)

    # Two nearest template descriptors for every scene descriptor
    order = np.argsort(d, axis=1)
    best_template = order[:, 0]
    d1 = d[np.arange(len(desc_scene)), order[:, 0]]
    d2 = d[np.arange(len(desc_scene)), order[:, 1]]

    # Lowe's ratio test
    keep = d1 < ratio * d2

    # Mutual nearest-neighbour check:
    # each selected template descriptor must also choose this scene point
    best_scene_for_template = np.argmin(d, axis=0)
    scene_indices = np.arange(len(desc_scene))
    mutual = best_scene_for_template[best_template] == scene_indices
    keep &= mutual

    scene_indices = np.flatnonzero(keep)
    if len(scene_indices) == 0:
        return np.zeros((0, 2), dtype=int)

    matches = np.column_stack(
        [best_template[scene_indices], scene_indices]
    ).astype(int)

    # Sort by descriptor distance, best match first
    return matches[np.argsort(d1[scene_indices])]

    # return np.zeros((0, 2), dtype=int)  # comment this line and write your code for the function


def predict_centres(kp_template: np.ndarray, kp_scene: np.ndarray,
                    matches: np.ndarray, ori_template: np.ndarray,
                    ori_scene: np.ndarray, template_shape: tuple) -> np.ndarray:
    """Turn every single match into a full guess at where Waldo is.

    This is the idea the whole task turns on. A matched keypoint carries a
    scale and an orientation as well as a position, so comparing them gives
    the size change and the rotation -- and once you know those, ONE match is
    enough to point at the centre of the whole object.

    For a match pairing template keypoint (y_t, x_t, sigma_t) at angle
    theta_t with scene keypoint (y_s, x_s, sigma_s) at theta_s, and template
    shape (H_t, W_t):

        s      = sigma_s / sigma_t                     the size change
        dtheta = theta_s - theta_t                     the rotation, degrees
        ox, oy = W_t / 2 - x_t,  H_t / 2 - y_t         offset to the centre
        cx     = x_s + s * (cos(dtheta) * ox - sin(dtheta) * oy)
        cy     = y_s + s * (sin(dtheta) * ox + cos(dtheta) * oy)

    A correct match puts (cx, cy) on Waldo. A wrong one puts it anywhere at
    all, possibly off the image. That is expected, and Task 4.4 cleans it up.

    Parameters
    ----------
    kp_template, kp_scene : np.ndarray
        Shapes (Nt, 3) and (Ns, 3), columns (y, x, sigma)
    matches : np.ndarray
        Shape (M, 2) from match_ratio_test
    ori_template, ori_scene : np.ndarray
        Orientations in degrees, one per keypoint
    template_shape : tuple
        (H_t, W_t) of the template these keypoints came from

    Returns
    -------
    np.ndarray
        Shape (M, 3), columns (cx, cy, s), in scene pixels. (0, 3) if no
        matches. Vectorise it -- no Python loop over matches.
    """
    M=np.asarray(matches,int).reshape(-1,2)
    if len(M)==0: return np.zeros((0,3))
    t=kp_template[M[:,0]]; s=kp_scene[M[:,1]]; ot=ori_template[M[:,0]]; os=ori_scene[M[:,1]]; scale=s[:,2]/np.maximum(t[:,2],1e-12); a=np.deg2rad(os-ot); ca,sa=np.cos(a),np.sin(a); th,tw=template_shape[:2]; ox,oy=tw/2-t[:,1],th/2-t[:,0]; cx=s[:,1]+scale*(ca*ox-sa*oy); cy=s[:,0]+scale*(sa*ox+ca*oy); return np.column_stack([cx,cy,scale])

    # return np.zeros((0, 3))  # comment this line and write your code for the function


def consensus_centre(predictions: np.ndarray, tol: float = 12.0,
                     min_votes: int = 8) -> tuple:
    """Find where the guesses pile up -- the DENSEST cluster, not the average.

    Only about one match in eight is correct here. Any method that assumes
    the good ones are a majority will fail, and that includes taking the
    median: the median of all the predictions lands in the middle of the
    scattered wrong ones and points at nothing. Measure that for yourself,
    it is one of the questions.

    Instead:
      1. Compute the distance from every prediction to every other one.
      2. Count, for each prediction, how many lie strictly within tol of it.
      3. The one with the highest count is the seed.
      4. The consensus is every prediction strictly within tol of that seed.
      5. Fewer than min_votes in it means no detection.
      6. Otherwise report the MEDIAN centre and MEDIAN scale of that cluster.

    Parameters
    ----------
    predictions : np.ndarray
        Shape (M, 3) from predict_centres
    tol : float
        How close two guesses must be to count as agreeing, in pixels
    min_votes : int
        Fewest agreeing guesses that counts as a detection

    Returns
    -------
    tuple
        centre (cx, cy) or None, scale (float) or None, and a boolean mask of
        shape (M,) marking which predictions joined the consensus
    """
    p=np.asarray(predictions,float).reshape(-1,3)
    if len(p)==0: return None,None,np.zeros(0,bool)
    d=p[:,None,:2]-p[None,:,:2]; dist=np.sqrt(np.sum(d*d,axis=2)); seed=int(np.argmax(np.sum(dist<tol,axis=1))); mask=dist[seed]<tol
    if mask.sum()<min_votes: return None,None,mask
    c=np.median(p[mask,:2],axis=0); return (float(c[0]),float(c[1])),float(np.median(p[mask,2])),mask

    # return None, None, np.zeros(0, dtype=bool)  # comment this line and write your code for the function


def find_waldo(scene: np.ndarray, template: np.ndarray,
               upsample: float = 2.0) -> dict:
    """Put it all together: detect, describe, match, vote.

    Parameters
    ----------
    scene : np.ndarray
        Grayscale scene
    template : np.ndarray
        Grayscale template at its native size
    upsample : float
        How much to enlarge the template before detecting

    Returns
    -------
    dict
        Keys: "box" ((x, y, w, h) or None), "n_matches", "n_votes",
        "matches", "votes", "kp_template", "kp_scene". Return None for the
        box when there is no consensus -- do not invent one.
    """
    scene=np.asarray(scene,float); template=np.asarray(template,float)
    h=max(4,int(round(template.shape[0]*upsample))); w=max(4,int(round(template.shape[1]*upsample)))
    t=cv2.resize(template,(w,h),interpolation=cv2.INTER_LINEAR) if upsample!=1 else template.copy()
    kt=dog_keypoints(t,sigma0=.6,n_scales=10,k=1.2,threshold=.003,max_keypoints=TEMPLATE_KEYPOINTS); ks=dog_keypoints(scene,sigma0=.6,n_scales=10,k=1.2,threshold=.003,max_keypoints=SCENE_KEYPOINTS)
    ot=np.array([dominant_orientation(t,y,x,s) for y,x,s in kt]); os=np.array([dominant_orientation(scene,y,x,s) for y,x,s in ks]); dt=describe_keypoints(t,kt,ot); ds=describe_keypoints(scene,ks,os); M=match_ratio_test(ds,dt,.8); P=predict_centres(kt,ks,M,ot,os,t.shape); c,s,v=consensus_centre(P,tol=20, min_votes=5) 
    
    box = None if c is None else box_from_centre(
        c[0], c[1], s, t.shape
    )

    return {'box':box,'n_matches':len(M),'n_votes':int(v.sum()),'matches':M,'votes':v,'kp_template':kt,'kp_scene':ks}

    # return {"box": None, "n_matches": 0, "n_votes": 0,
    #         "matches": np.zeros((0, 2), int),
    #         "votes": np.zeros(0, bool),
    #         "kp_template": np.zeros((0, 3)),
    #         "kp_scene": np.zeros((0, 3))}  # comment this line and write your code for the function


if __name__ == "__main__":

    # The two pipelines want different inputs, and the difference is the
    # point. Correlation (Task 2) keeps all three channels, because Waldo's
    # red is most of what identifies him. The feature pipeline (Task 3)
    # needs a scalar field to take gradients of, so it gets luminance.
    template_rgb = load_image_as_rgb("imgs/waldo_template.png")
    template = to_grayscale(template_rgb)
    scene_names = ["scene_easy", "scene_medium", "scene_rotated", "scene_hard"]

    print("{:<16}{:>9}{:>8}{:>9}{:>10}".format(
        "scene", "matches", "votes", "IoU", "agree?"))
    for name in scene_names:
        scene_rgb = load_image_as_rgb("imgs/{}.png".format(name))
        scene = to_grayscale(scene_rgb)
        truth = load_ground_truth(name)

        # ###############################################
        # Comment these lines and write your code here
        # found = find_waldo(scene, template)
        # correlation = detect_multiscale(scene_rgb, template_rgb)
        found = find_waldo(scene, template)
        correlation = detect_multiscale(scene_rgb, template_rgb)
        # ###############################################

        feature_box = found["box"]
        corr_box = None
        if correlation:
            best = correlation[0]
            corr_box = (best["x"], best["y"], best["w"], best["h"])

        # Scored against the truth here, for your report only. Notice that
        # find_waldo itself never sees it.
        overlap = iou(feature_box, truth) if feature_box else 0.0
        # Do the two completely different methods agree with each other?
        agreement = (iou(feature_box, corr_box)
                     if feature_box and corr_box else 0.0)
        print("{:<16}{:>9}{:>8}{:>9.3f}{:>10.3f}".format(
            name, found["n_matches"], found["n_votes"], overlap, agreement))

        draw_matches(template, scene_rgb, found["kp_template"],
                        found["kp_scene"], found["matches"], found["votes"],
                        f"plots/task4_{name}_matches.png")

        shown = draw_box(scene_rgb, truth, (0.1, 0.4, 1.0))
        if feature_box:
            shown = draw_box(shown, feature_box,
                             (0, 1, 0) if overlap >= 0.5 else (1, 0, 0))
        plt.figure(figsize=(9, 7))
        plt.imshow(shown)
        plt.title("{}: blue = truth, green/red = your detection".format(name))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("plots/task4_{}.png".format(name),
                    metadata={"Author": getpass.getuser()})
        plt.close()

    print("\nwrote plots/task4_matches.png and plots/task4_<scene>.png")
    print("The last column is agreement between Task 2 and Task 4, computed")
    print("without any ground truth. Compare it against the IoU column.")
