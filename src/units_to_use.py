import numpy as np
import os


units_to_use = {
    "BroGre4497M_180609_105949": [
        (8, 129, [(27000, np.inf)]),
        (8, 135, [(11000, 25000)]),
        (8, 136, [(0, np.inf)]),  # multiunit
        (8, 138, [(0, 4000)]),    # multiunit
        (8, 140, [(0, 13000)]),
        (10, 31, [(0, 6000), (15000, 23000)]),
        (10, 32, [(0, np.inf)]),
        (11, 197, [(5000, 10000), (20000, 25000)]),  # precise but possibly different between BOS and tutor of the same syllable
        (11, 198, [(8000, 25000)]),         # another interesting one, possibly more suppressed around peaks in bos live vs tutor live
        (11, 200, [(11000, 20000)]),        # high firing, dont see much but it definitly does something during bos vs tutor live
        (11, 201, [(0, 11000), (20000, np.inf)]),  # likes the long syllable in playbacks, responses similarly in tutor and bos maybe but not exactly same
        (12, 17, [(0, np.inf)]),
        (12, 18, [(0, np.inf)]),  # very different beweetn bos and tutor live, tutor live and pb identical
        (13, 180, [(0, 9000)]),  # gain possibly different beween bos and tutor
        (13, 190, [(3500, 4500)]),
        (13, 192, [(500, 4500)]),
        (13, 193, [(7000, 9000)]),
        (13, 194, [(5250, 5750)]),    # looks similar to other clusters (may want to merge), PB, live tutor, live BOS
        (13, 195, [(0, 3000)]), 
        (13, 196, [(3500, 9000)]),  # has similar respons in bos vs tutor live but it is incluear if it shows up in playbacks
        (13, 197, [(0, 9000)]),
        (14, 2, [(0, np.inf)]),
        (16, 59, [(0, np.inf)]),
        (16, 60, [(0, np.inf)]),
        (16, 61, [(500, 2000)]),    # looks similar to other clusters (may want to merge), PB, live tutor, live BOS
        (16, 62, [(0, 20000)]), 
        (21, 71, [(8000, 15000)]),  # didnt overlap playbacks very much
        (21, 72, [(0, 3000)]),
        (21, 74, [(11000, 19000)]), # good strf!!!!!!!!!!! and very different tutor vs bo
        (21, 75, [(0, np.inf)]),    # decent strf!!!!!!!!! and very differnet tutor vs bos
    ],

    "BroGre4497M_180610_114535": [
        (10, 34, [(0, 1200)]),    # well isolated but pb only
        (10, 39, [(0, 5000)]),    # bad, changes too much late to be able to tell... but could compare earlier tut live to playbacks
        (11, 1, [(0, 8000)]),   # *** bos and tutor appears different
        (12, 17, [(0, 3000)]),
        (12, 18, [(4000, np.inf)]),  # ******* BOS AND TUTOR VERY DIFEERN AND BNOS AND LIVE VERY DIFFERENT, compare to second half of playbacks? ... but need to re-split this one up
        (14, 17, [(0, np.inf)]),  # **? also different bos vs tutor, and looks pretty diferent from playbacks
        (14, 18, [(0, np.inf)]),  # very similar to 14 17 mayben merge them
        (14, 22, [(0, np.inf)]),  # also simialr to other 14 ones
        (16, 36, [(0, 3600), (3600, np.inf)]), # higher firing rate to tutor live than bos
        (16, 37, [(0, np.inf)]),
        (16, 38, [(0, np.inf)]),  # *** ? but doesnt look auditory
        (21, 3, [(0, np.inf)]),  # looks imliar to 21 4, different across tutor live and bos live
        (21, 4, [(0, np.inf)]),     
        (21, 5, [(0, 9000)]),   # sparse and selecive for spicfic times in song. might be interseting. has a reliable bos peak thats not in tutor live
        (23, 26, [(6000, np.inf)]),
        (23, 27, [(0, np.inf)]),  # similar responses bos and tutor
        (23, 28, [(0, np.inf)]),  # simlar responses bos and tutor
    ],

    "GreYel3594M___170915_082724": [
        (9, 1, [(13000, 15000), (18000, 20000)]),  # split up, noisy recording, PB only
        (12, 66, [(0, 8000)]),                     # SU, good wf shape, PB only
        (12, 80, [(17500, 25000)]),                 # SU+, well isolated, PB, live BOS
        (12, 82, [(0, 5000)]),                      # SU, similar to 80 at diff time, PB only
        (13, 76, [(1000, 4000)]),    # well isolated but pb only
        # (13, 87, [(0, 18000)]), # relaly isnt around very long
        (13, 94, [(10000, 30000)]),  # nice response might be wroth cutting off at 15000
        (13, 104, [(32000, 39000)]),  # same unit as above but way better isolated
        (14, 60, [(0, 4000)]),
        (14, 61, [(0, 25000)]),    # high firing response (maybe because multiunit) to BOS and TUT
        (14, 62, [(20000, 30000)]),
        (15, 1, [(0, 20000)]),
        (21, 31, [(0, 3000)]),
        (21, 47, [(0, 19000)]),
        (22, 57, [(0, 2000), (2500, 5000)]),
        (22, 71, [(2500, 7000)]),
        (22, 77, [(0, 19000)]),
        (22, 78, [(0, 2500)]), 
        (23, 40, [(0, 7000)]),
        (23, 42, [(0, 2500)]),
        (23, 54, [(0, 5000)]),
        (23, 56, [(0, 16000)]),
    ],

    "GreYel3594M___170919_094646": [
        (8, 1, [(0, np.inf)]), 
        (9, 1, [(0, np.inf)]), 
        (11, 2, [(0, np.inf)]),                     
        (12, 21, [(1000, 2500)]),                
        (12, 23, [(4000, 7000)]),                     
        (12, 59, [(8500, 12000)]),    
        (12, 61, [(0, np.inf)]),
        (13, 36, [(2500, 10000)]),
        (13, 37, [(0, 13000)]),
        (14, 1, [(0, np.inf)]),
        (15, 1, [(0, np.inf)]),
        (18, 21, [(0, np.inf)]),
        (18, 22, [(2500, 8000)]),
        (19, 29, [(11000, 13000)]),
        (19, 30, [(0, np.inf)]),
        (21, 24, [(0, 7000)]),
        (21, 25, [(0, np.inf)]),
        (22, 2, [(0, np.inf)]),
        (23, 19, [(1000, 4000)]), 
        (23, 20, [(0, np.inf)]),
    ],

    "RedGre7705M__171214_102641": [
        (8, 43, [(1000, 14000)]),
        (8, 44, [(0, np.inf)]), 
        (15, 46, [(0, 800)]),                     
        (15, 61, [(5000, 9000)]),    # extremeley precise, probably L
        (15, 62, [(1000, np.inf)]),  # same tutor live and pb responses
        (22, 26, [(0, np.inf)]),     # stereotyped response in tutor live, not in tutor pb
        (22, 27, [(0, np.inf)]),     # another baisc one, probably L
    ],

    "RedGre7705M__171214_145611": [
        (15, 1, [(0, np.inf)]), 
        (22, 1, [(0, np.inf)]),
        (22, 4, [(0, 2000)]),                     
        (22, 5, [(0, np.inf)]),    
    ],

    "RedGre7705M__171217_142728": [
        (10, 1, [(0, np.inf)]), 
        (15, 1, [(500, 2000)]),                     
        (21, 1, [(0, np.inf)]),    
        (22, 50, [(0, np.inf)]),
        (22, 51, [(0, np.inf)]),
    ],

    "LbY6074__161215_094456": [
        (12, 28, [(12000, 28000)]),   # these two might be pretty much the same... consider merging
        (12, 29, [(10000, 25000)]),                     
        (12, 30, [(10000, 30000)]),    
        (12, 31, [(33000, np.inf)]),     # only su?
        (13, 1, [(8000, 30000)]),
        (15, 60, [(13000, np.inf)]),  # merge with below?
        (15, 61, [(10000, 29000)]),  # SU
        (20, 2, [(13000, 18000)]),  # SU
        (21, 57, [(9000, np.inf)]), #MU
        (21, 62, [(9000, np.inf)]), #MU
        (21, 65, [(11000, 29000)]), #SU
        (21, 69, [(9000, 35000)]), #SU
        (22, 26, [(9000, np.inf)]), #MU
        (22, 28, [(28000, np.inf)]), #SU
        (22, 29, [(9000, np.inf)]), #MU
    ],

    "LBlYel0923___160901_103806": [
        (8, 4, [(0, 5000)]),  # MU
        (8, 5, [(6000, np.inf)]), #MU
        (9, 67, [(0, 9000)]),     # SU *
        (9, 68, [(0, np.inf)]),     # SU  *
        (9, 70, [(0, np.inf)]),    # MU bad
        (12, 70, [(0, 10000)]),  # SU * 
        (12, 71, [(10000, np.inf)]),  # SU * essentially continuation of previous
        (14, 16, [(0, 12000)]),  # MU + 
        (14, 20, [(0, np.inf)]), # MU +
        (17, 1, [(0, np.inf)]), #MU + 
        (19, 1, [(0, np.inf)]), # SU +
        (20, 1, [(0, np.inf)]), # MU
        (21, 1, [(0, np.inf)]), # MU
        (22, 1, [(0, np.inf)]), # MU
    ],
}


channel_map = {
    "BroGre4497M_180610_114535": 1,
    "BroGre4497M_180609_105949": 1,
    "GreYel3594M___170915_082724": 0,
    "GreYel3594M___170919_094646": 0,
    "RedGre7705M__171214_102641": 0,
    "RedGre7705M__171214_145611": 0,
    "RedGre7705M__171217_142728": 0,
    "LbY6074__161215_094456": 0,
    "LBlYel0923___160901_103806": 0
}


NCM = ("NCM",)
CM = ("CM",)
L = ("L",)
CM_L = ("CM", "L")
NCM_L = ("NCM", "L")
NON = tuple()


putative_brain_regions = {
    "BroGre4497M_180609_105949": {
        (8, 129): NON,
        (8, 135): NON,
        (8, 136): NON,
        (8, 138): NON,
        (8, 140): NON,
        (10, 31): NCM_L,
        (10, 32): NCM_L,  # * new
        (11, 197): L,
        (11, 198): L,
        (11, 200): L,
        (11, 201): L,
        (12, 17): NCM,
        (12, 18): NCM,
        (13, 180): L,
        (13, 190): L,
        (13, 192): L,
        (13, 193): L,
        (13, 194): L,
        (13, 195): L,
        (13, 196): L,
        (13, 197): L,
        (14, 2): CM,
        (16, 59): NCM,
        (16, 60): NCM,
        (16, 61): NCM,
        (16, 62): NCM,
        (21, 71): L,
        (21, 72): L,
        (21, 74): L,
        (21, 75): L,
    },

    "BroGre4497M_180610_114535": {
        (10, 34): NCM,
        (10, 39): NCM,
        (11, 1): NCM,
        (12, 17): L,
        (12, 18): L,
        (14, 17): CM_L,
        (14, 18): CM_L,
        (14, 22): CM_L,
        (16, 36): NCM,
        (16, 37): NCM,
        (16, 38): NCM,
        (21, 3): L,
        (21, 4): L,
        (21, 5): L,
        (23, 26): CM_L,
        (23, 27): CM_L,
        (23, 28): CM_L,
    },

    "GreYel3594M___170915_082724": {
        (9, 1): CM,  # it says caudal nido
        (12, 66): CM,
        (12, 80): CM,
        (12, 82): CM,
        (13, 76): CM,
        (13, 94): CM,
        (13, 104): CM,
        (14, 60): L,
        (14, 61): L,
        (14, 62): L,
        (15, 1): NCM_L,
        (21, 31): CM,
        (21, 47): CM,
        (22, 57): L,
        (22, 71): L,
        (22, 77): L,
        (22, 78): L,
        (23, 40): L,  # * L2b
        (23, 42): L,
        (23, 54): L,
        (23, 56): L,
    },

    "GreYel3594M___170919_094646": {
        (8, 1): NON,  # it says caudal nido
        (9, 1): CM,
        (11, 2): CM,
        (12, 21): CM,
        (12, 23): CM,
        (12, 59): CM,
        (12, 61): CM,
        (13, 36): L,
        (13, 37): L,
        (14, 1): L,
        (15, 1): NCM,
        (18, 21): CM,
        (18, 22): CM,
        (19, 29): CM,
        (19, 30): CM,
        (21, 24): L,
        (21, 25): L,
        (22, 2): L,
        (23, 19): NCM,  # *uhhhhh it says L and NCM
        (23, 20): NCM,
    },

    "RedGre7705M__171214_102641": {
        (8, 43): NCM,
        (8, 44): NCM,
        (15, 46): L,
        (15, 61): L,
        (15, 62): L,
        (22, 26): CM_L,
        (22, 27): CM_L,
    },

    "RedGre7705M__171214_145611": {
        (15, 1): CM_L,
        (22, 1): CM_L,
        (22, 4): CM_L,
        (22, 5): CM_L,
    },

    "RedGre7705M__171217_142728": {
        (10, 1): NCM,
        (15, 1): CM_L,
        (21, 1): L,
        (22, 50): CM_L,
        (22, 51): CM_L,
    },

    "LbY6074__161215_094456": {
        (12, 28): NON,
        (12, 29): NON,
        (12, 30): NON,
        (12, 31): NON,
        (13, 1): NON,
        (15, 60): NON,
        (15, 61): NON,
        (20, 2): NON,
        (21, 57): NON,
        (21, 62): NON,
        (21, 65): NON,
        (21, 69): NON,
        (22, 26): NON,
        (22, 28): NON,
        (22, 29): NON,
    },

    "LBlYel0923___160901_103806": {
        (8, 4): NON,
        (8, 5): NON,
        (9, 67): NON,
        (9, 68): NON,
        (9, 70): NON,
        (12, 70): NON,
        (12, 71): NON,
        (14, 16): NON,
        (14, 20): NON,
        (17, 1): NON,
        (19, 1): NON,
        (20, 1): NON,
        (21, 1): NON,
        (22, 1): NON,
    },
}


is_single_unit = {
    "BroGre4497M_180609_105949": {
        (8, 129): 1,
        (8, 135): 1,
        (8, 136): 0,
        (8, 138): 1,
        (8, 140): 1,
        (10, 31): 1,
        (10, 32): 0,  # * new
        (11, 197): 1,
        (11, 198): 1,
        (11, 200): 1,
        (11, 201): 0,
        (12, 17): 1,
        (12, 18): 1,
        (13, 180): 1,
        (13, 190): 1,
        (13, 192): 1,
        (13, 193): 1,
        (13, 194): 1,
        (13, 195): 1,
        (13, 196): 1,
        (13, 197): 0,
        (14, 2): 0,
        (16, 59): 1,
        (16, 60): 0,
        (16, 61): 1,
        (16, 62): 1,
        (21, 71): 1,
        (21, 72): 1,
        (21, 74): 1,
        (21, 75): 0,
    },

    "BroGre4497M_180610_114535": {
        (10, 34): 1,
        (10, 39): 0,
        (11, 1): 1,
        (12, 17): 1,
        (12, 18): 0,
        (14, 17): 0,
        (14, 18): 0,
        (14, 22): 0,
        (16, 36): 0,
        (16, 37): 1,
        (16, 38): 1,
        (21, 3): 0,
        (21, 4): 1,
        (21, 5): 1,
        (23, 26): 1,
        (23, 27): 1,
        (23, 28): 0,
    },

    "GreYel3594M___170915_082724": {
        (9, 1): 1,  # it says caudal nido
        (12, 66): 1,
        (12, 80): 1,
        (12, 82): 1,
        (13, 76): 1,
        (13, 94): 0,
        (13, 104): 1,
        (14, 60): 1,
        (14, 61): 1,
        (14, 62): 1,
        (15, 1): 1,
        (21, 31): 1,
        (21, 47): 1,
        (22, 57): 1,
        (22, 71): 1,
        (22, 77): 0,
        (22, 78): 1,
        (23, 40): 1,  # * L2b
        (23, 42): 1,
        (23, 54): 0,
        (23, 56): 0,
    },

    "GreYel3594M___170919_094646": {
        (8, 1): 0,  # it says caudal nido
        (9, 1): 0,
        (11, 2): 0,
        (12, 21): 1,
        (12, 23): 1,
        (12, 59): 1,
        (12, 61): 0,
        (13, 36): 1,
        (13, 37): 0,
        (14, 1): 0,
        (15, 1): 0,
        (18, 21): 0,
        (18, 22): 0,
        (19, 29): 1,
        (19, 30): 0,
        (21, 24): 1,
        (21, 25): 0,
        (22, 2): 0,
        (23, 19): 1,  # *uhhhhh it says L and NCM
        (23, 20): 0,
    },

    "RedGre7705M__171214_102641": {
        (8, 43): 1,
        (8, 44): 1,
        (15, 46): 1,
        (15, 61): 1,
        (15, 62): 0,
        (22, 26): 1,
        (22, 27): 1,
    },

    "RedGre7705M__171214_145611": {
        (15, 1): 0,
        (22, 1): 0,
        (22, 4): 1,
        (22, 5): 0,
    },

    "RedGre7705M__171217_142728": {
        (10, 1): 0,
        (15, 1): 0,
        (21, 1): 0,
        (22, 50): 1,
        (22, 51): 0,
    },

    "LbY6074__161215_094456": {
        (12, 28): 0,
        (12, 29): 0,
        (12, 30): 0,
        (12, 31): 1,
        (13, 1): 0,
        (15, 60): 0,
        (15, 61): 1,
        (20, 2): 1,
        (21, 57): 0,
        (21, 62): 0,
        (21, 65): 1,
        (21, 69): 1,
        (22, 26): 0,
        (22, 28): 1,
        (22, 29): 0,
    },

    "LBlYel0923___160901_103806": {
        (8, 4): 0,  # MU
        (8, 5): 0,  #MU
        (9, 67): 1,  # SU *
        (9, 68): 1,  # SU  *
        (9, 70): 0,  # MU bad
        (12, 70): 1,  # SU * 
        (12, 71): 1,  # SU * essentially continuation of previous
        (14, 16): 0,  # MU
        (14, 20): 0,  # MU
        (17, 1): 0,  #MU + 
        (19, 1): 1,  # SU +
        (20, 1): 0,  # MU
        (21, 1): 0,  # MU
        (22, 1): 0,  # MU
    },
}


def custom_correction(sound_events, site):
    for e in sound_events:
        if e["sound_type"] == "playback":
            if "_So" in e["playback_file"]:
                e["vocalization_type"] = "So"

        if e["sound_type"] == "live" and "GreYel3594M___" in site:
            if e["vocalizer"] == "tutor":
                e["vocalizer"] = "con_male_familiar"

        if e["sound_type"] == "playback" and "GreYel3594M___" in e["playback_file"]:
            pb_file = os.path.basename(e["playback_file"])
            if "MAT" in pb_file:
                e["vocalizer"] = "con_male_familiar"
            elif "MAF" in pb_file:
                e["vocalizer"] = "tutor"
            elif "MJF" in pb_file:
                e["vocalizer"] = "juvenile"

        if e["sound_type"] == "playback" and "RedGre7705M__" in e["playback_file"]:
            pb_file = os.path.basename(e["playback_file"])
            if "MJP" in pb_file:
                e["vocalizer"] = "subject"
            elif e["vocalizer"] == "con_male":
                e["vocalizer"] = "con_male_familiar"

        if e["sound_type"] == "playback" and "BroGre4497M_" in e["playback_file"]:
            pb_file = os.path.basename(e["playback_file"])
            if "MJB" in pb_file:
                e["vocalizer"] = "juvenile"
            elif "MAF" in pb_file:
                e["vocalizer"] = "con_male_familiar"

        if e["sound_type"] == "playback" and ("LBlYel0923_" in e["playback_file"] or "LblYel0923_" in e["playback_file"]):
            pb_file = os.path.basename(e["playback_file"])
            if "MAF" in pb_file:
                e["vocalizer"] = "con_male_familiar"

        if e["sound_type"] == "playback" and "LbY6074_" in e["playback_file"]:
            pb_file = os.path.basename(e["playback_file"])
            if "MAF" in pb_file:
                e["vocalizer"] = "con_male_familiar"

    return sound_events

