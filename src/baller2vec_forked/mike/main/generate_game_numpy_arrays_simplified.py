
import os 

from baller2vec_forked.settings import GAMES_DIR, TRACKING_DIR

from baller2vec_forked.mike.make_data import (
    get_playerid2player_idx_map,
    get_shot_times, 
    get_team_hoop_sides,
    get_event_streams, 
    save_numpy_arrays, 
    save_baller2vec_config,
)


"""
We run a simple verison of generate_game_numpy_arrays.py from baller2vec,
so that we can 

1) provide comments on what is happening (what data is grabbed, what processing is done).
2) control the size of the dataset processed (in particular we can try to apply transformer on a small snippet)
3) run sequentially rather than in parallel.  (slower OFC but allows interactivity/greater visibility)
4) and run locally (at least to start. can move to cloud later)

"""

### Preliminiaries
os.makedirs(GAMES_DIR, exist_ok=True)

### Run simple version of get_playerid2player_idx_map().
# "Simple" = sequential not parallel.  Perhaps also fewer games.
# That parallel script calls playerid2player_idx_map_worker().

# `all_game_7zs` gives a list of filenames, each one giving a game, 
# e.g.  '01.07.2016.ATL.at.PHI.7z'.  It seems that these are contained in 
# `NBA-Player-Movements/data/`.  Locally they are contained in 
# `baller2vec_forked/data//NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs' 
#  There are 636 games.
dir_fs = os.listdir(TRACKING_DIR)
all_game_7zs = [dir_f for dir_f in dir_fs if dir_f.endswith(".7z")]

#sample_games_7zs = all_game_7zs[:N_GAMES]
#sample_games_7zs=[x for x in all_game_7zs if "TOR" in x and "CHA" in x]
#sample_games_7zs=[x for x in all_game_7zs if "TOR" in x]
sample_games_7zs=['01.01.2016.CHA.at.TOR.7z']

# TODO: Split this into two functions, one that unzips all the files first, and then 
# one that gets the player indices

# TODO: Ten of the filenames are off in the source data at 
# https://github.com/linouk23/NBA-Player-Movements/tree/master/data/2016.NBA.Raw.SportVU.Game.Logs: 
# e.g.'2016.NBA.Raw.SportVU.Game.Logs12.05.2015.NYK.at.MIL.7z' rather than
#  '10.31.2015.GSW.at.NOP.7z'.  The former appears to have the higher-level folder
#  directory accidentally appened to the basename.  Dealing with the dual naming syntax can cause weirdnesses
# throughout. Should I fix this?  Probably would be best to just find and remove the prefixes up front.
(playerid2player_idx, player_idx2props) = get_playerid2player_idx_map(sample_games_7zs)


shot_times = get_shot_times(sample_games_7zs)
# TODO: Why is get_team_hoop_sides so much slower than get_shot_times and get_event_streams? Can I speed it up?
hoop_sides = get_team_hoop_sides(sample_games_7zs, shot_times)
(event2event_idx, gameid2event_stream) = get_event_streams(sample_games_7zs)

save_numpy_arrays(sample_games_7zs, gameid2event_stream,  hoop_sides, event2event_idx, playerid2player_idx) 
save_baller2vec_config(player_idx2props, event2event_idx)
