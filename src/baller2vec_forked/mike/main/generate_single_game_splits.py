import os 

from baller2vec_forked.mike.make_data import (
    get_playerid2player_idx_map,
    get_player_idx2playing_time_map,
    get_shot_times, 
    get_team_hoop_sides,
    get_event_streams, 
    save_game_numpy_arrays, 
    save_baller2vec_info,
    _gameid_from_game7z_basename
)

### 
# MAIN
###

### Configs
INFO_DIR = "/Users/mwojno01/Repos/baller2vec_forked/data/single_game_splits/info"
SPLITS_DIR = "/Users/mwojno01/Repos/baller2vec_forked/data/single_game_splits/splits"

### Code 

os.makedirs(SPLITS_DIR, exist_ok=True)

game_name = '01.01.2016.CHA.at.TOR.7z'
gameid = _gameid_from_game7z_basename(game_name)

(event2event_idx, gameid2event_stream) = get_event_streams([game_name])
event_stream_for_one_game=list(gameid2event_stream.values())[0]
(playerid2player_idx, player_idx2props) = get_playerid2player_idx_map([game_name])
shot_times = get_shot_times([game_name])
# TODO: Why is get_team_hoop_sides so much slower than get_shot_times and get_event_streams? Can I speed it up?
hoop_sides = get_team_hoop_sides([game_name], shot_times)
(event2event_idx, gameid2event_stream) = get_event_streams([game_name])


event_stream_for_one_game = gameid2event_stream[gameid]

# train data
save_game_numpy_arrays(
    game_name, event_stream_for_one_game,  hoop_sides, event2event_idx, playerid2player_idx, SPLITS_DIR, first_event_idx=0,
    last_event_idx=3, 
) 
# val data
save_game_numpy_arrays(
    game_name, event_stream_for_one_game,  hoop_sides, event2event_idx, playerid2player_idx, SPLITS_DIR, first_event_idx=4,
    last_event_idx=4, 
) 
# test data
save_game_numpy_arrays(
    game_name, event_stream_for_one_game,  hoop_sides, event2event_idx, playerid2player_idx, SPLITS_DIR, first_event_idx=5,
    last_event_idx=5, 
) 

player_idx2playing_time = get_player_idx2playing_time_map(SPLITS_DIR)
save_baller2vec_info(player_idx2props, player_idx2playing_time, event2event_idx, INFO_DIR)


