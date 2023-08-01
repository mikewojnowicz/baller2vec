import glob
from typing import List, Dict, Optional, Tuple 
import os 
import pickle 
import py7zr
import shutil

import numpy as np
import pandas as pd

from baller2vec_forked.settings import (
    TRACKING_DIR, EVENTS_DIR, DATA_DIR, COURT_LENGTH, TEAM_ID2PROPS
)

from baller2vec_forked.generate_game_numpy_arrays import (
    add_score_changes, 
    get_event_stream, 
    get_game_hoop_sides, 
)


### 
# Configs
###

HALF_COURT_LENGTH = COURT_LENGTH // 2
WALL_CLOCK_DIFF_THRESHOLD = 1.0

###
# Players 
###

def get_playerid2player_idx_map(game_7z_filenames : List[str]) -> Tuple[Dict, Dict]:
    """
    Partial example of first return value (the dictionary `playerid2player_idx`):
        {201564: 0,
        203487: 1,
        203142: 2,
        203898: 3,
        203953: 4,

    Partial example of second return value (the dictionary `player_idx2props`):
        {0: {'name': 'O.J. Mayo', 'playerid': 201564},
        1: {'name': 'Michael Carter-Williams', 'playerid': 203487},
        2: {'name': 'Chris Copeland', 'playerid': 203142},
        3: {'name': 'Tyler Ennis', 'playerid': 203898},
        4: {'name': 'Jabari Parker', 'playerid': 203953},
        5: {'name': 'Greg Monroe', 'playerid': 202328},
        6: {'name': 'Damien Inglis', 'playerid': 203996},

    """
    playerid2player_idx = {}
    player_idx2props = {}

    _playerid2props={}

    # TODO: Parallelize 
    for game_7z in game_7z_filenames:
        game_name = game_7z.split(".7z")[0]

        try:
            # this converts the .7z archive file into a json file in the same folder 
            archive = py7zr.SevenZipFile(f"{TRACKING_DIR}/{game_7z}", mode="r")
            archive.extractall(path=f"{TRACKING_DIR}/{game_name}")
            archive.close()
        except AttributeError:
            print(f"{game_name}\nBusted.", flush=True)
            shutil.rmtree(f"{TRACKING_DIR}/{game_name}")
            continue 

        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        except IndexError:
            print(f"No tracking data for {game_name}.", flush=True)
            shutil.rmtree(f"{TRACKING_DIR}/{game_name}")
            continue 
    
        df_tracking = pd.read_json(f"{TRACKING_DIR}/{game_name}/{gameid}.json")
        events = df_tracking["events"].iloc[0]
        players = events["home"]["players"] + events["visitor"]["players"]

        for player in players:
            playerid = player["playerid"]
            _playerid2props[playerid] = {
                "name": " ".join([player["firstname"], player["lastname"]]),
            }

    # relabel the dictionaries in the way baller2vec does
    # `playerid2player_idx` is Dict[int,int] mapping player_id (e.g. 2570) -> player_idx (0,1,2,...)
    # ` player_idx2props` is Dict[int, Dict] mapping player_idx (0,1,2,..) to a dict that looks like
    # {'name': 'Marreese Speights', 'playerid': 201578}
    playerid2player_idx = {}
    player_idx2props = {}
    for (player_idx, playerid) in enumerate(_playerid2props):
        playerid2player_idx[playerid] = player_idx
        player_idx2props[player_idx] = _playerid2props[playerid]
        player_idx2props[player_idx]["playerid"] = playerid
    return  playerid2player_idx, player_idx2props

def get_player_idx2playing_time_map(games_dir: str) -> Dict[int, float]:
    """
    Partial example of return value:

        {0: 1683.7159999992375,
        4: 1730.3049999992004,
        5: 1622.953999999284,
        9: 1574.316999999348,
    """
    player_idx2playing_time = {}

    file_pattern = "*_X.npy"

    # Use glob to get a list of all files matching the pattern
    matching_files = glob.glob(f"{games_dir}/{file_pattern}")

    for filepath in matching_files:
        X = np.load(filepath)
        wall_clock_diffs = np.diff(X[:, -1]) / 1000
        all_player_idxs = X[:, 10:20].astype(int)
        prev_players = set(all_player_idxs[0])
        for (row_idx, player_idxs) in enumerate(all_player_idxs[1:]):
            current_players = set(player_idxs)
            if len(prev_players & current_players) == 10:
                wall_clock_diff = wall_clock_diffs[row_idx]
                if wall_clock_diff < WALL_CLOCK_DIFF_THRESHOLD:
                    for player_idx in current_players:
                        player_idx2playing_time[player_idx] = (
                            player_idx2playing_time.get(player_idx, 0) + wall_clock_diff
                        )

            prev_players = current_players

    return player_idx2playing_time


### 
# Shot times 
###

def get_game_time(game_clock_secs, period : int) -> int:
    """
    Shows the game time in seconds (represented as an integer). 
    There are 60*12=720 seconds per quarter, and so 2880 seconds per game. 
    """
    period_secs = 720 if period <= 4 else 300
    period_time = period_secs - game_clock_secs
    if period <= 4:
        return (period - 1) * 720 + period_time
    else:
        return 4 * 720 + (period - 5) * 300 + period_time
    

def get_shot_times(game_7z_filenames : List[str]) -> Dict[int, Dict[int,str]]:
    """
    Constructs dictionary mapping gameid to a dictionary of shot times, represented
    as game_time (in seconds, from 0 to 2880) -> str of team who took the shot

    Example return value:

     '0021500035': {21: 'NOP',
            30: 'GSW',
            53: 'NOP',
            68: 'GSW',
            80: 'NOP',
            86: 'NOP',
            123: 'NOP',
            132: 'GSW',
            138: 'NOP',
            146: 'GSW',
            166: 'NOP',
            173: 'GSW',
            178: 'NOP',
    """
    shot_times = {}
    for game_7z in game_7z_filenames:
        game_name = game_7z.split(".7z")[0]
        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        except FileNotFoundError:
            continue

        df_events = pd.read_csv(f"{EVENTS_DIR}/{gameid}.csv")
        game_shot_times = {}
        for (row_idx, row) in df_events.iterrows():
            period = row["PERIOD"]
            game_clock = row["PCTIMESTRING"].split(":")
            game_clock_secs = 60 * int(game_clock[0]) + int(game_clock[1])
            game_time = get_game_time(game_clock_secs, period)

            if (row["EVENTMSGTYPE"] == 1) or (row["EVENTMSGTYPE"] == 2):
                game_shot_times[game_time] = row["PLAYER1_TEAM_ABBREVIATION"]

        shot_times[gameid] = game_shot_times
    return shot_times 


### 
# Hoop sides
###

def get_team_hoop_sides(
    game_7z_filenames: List[str],
    shot_times,
) -> Dict[int, Dict[int, Dict[str,int]]]:
    """
    Returns a dict mapping gameid to another dict, which maps
    quarters to an innermost dict, which maps team strings to 
    the x-coordinate of the basket 
    (forced to be either 0 or 94; the COURT_LENGTH is defined to be 94
    in the settings)

    Partial example return value:
        {'0021500295': {1: {'MIL': 94, 'NYK': 0},
                        2: {'MIL': 94, 'NYK': 0},
                        3: {'MIL': 0, 'NYK': 94},
                        4: {'MIL': 0, 'NYK': 94}}}
    """
    hoop_sides = {}
    for game_7z in game_7z_filenames:
        game_name = game_7z.split(".7z")[0]

        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        except FileNotFoundError:
            continue

        df_tracking = pd.read_json(f"{TRACKING_DIR}/{game_name}/{gameid}.json")
        hoop_side_counts = {}
        used_game_times = set()
        teams = set()
        for tracking_event in df_tracking["events"]:
            for moment in tracking_event["moments"]:
                period = moment[0]
                game_clock = moment[2]
                game_time = int(get_game_time(game_clock, period))
                if (game_time in shot_times[gameid]) and (
                    game_time not in used_game_times
                ):
                    ball_x = moment[5][0][2]

                    if ball_x < HALF_COURT_LENGTH:
                        hoop_side = 0
                    else:
                        hoop_side = COURT_LENGTH

                    if period not in hoop_side_counts:
                        hoop_side_counts[period] = {}

                    shooting_team = shot_times[gameid][game_time]
                    if shooting_team not in hoop_side_counts[period]:
                        hoop_side_counts[period][shooting_team] = {
                            0: 0,
                            COURT_LENGTH: 0,
                        }

                    hoop_side_counts[period][shooting_team][hoop_side] += 1
                    used_game_times.add(game_time)
                    teams.add(shooting_team)

        if len(teams) == 0:
            print(f"The moments in the {game_name} JSON are empty.", flush=True)
            continue

        hoop_sides[gameid] = get_game_hoop_sides(teams, hoop_side_counts, game_name)
    return hoop_sides 

### 
# Shot times 
###

def get_game_time(game_clock_secs, period : int) -> int:
    """
    Shows the game time in seconds (represented as an integer). 
    There are 60*12=720 seconds per quarter, and so 2880 seconds per game. 
    """
    period_secs = 720 if period <= 4 else 300
    period_time = period_secs - game_clock_secs
    if period <= 4:
        return (period - 1) * 720 + period_time
    else:
        return 4 * 720 + (period - 5) * 300 + period_time
    

def get_shot_times(game_7z_filenames : List[str]) -> Dict[int, Dict[int,str]]:
    """
    Constructs dictionary mapping gameid to a dictionary of shot times, represented
    as game_time (in seconds, from 0 to 2880) -> str of team who took the shot

    Example return value:

     '0021500035': {21: 'NOP',
            30: 'GSW',
            53: 'NOP',
            68: 'GSW',
            80: 'NOP',
            86: 'NOP',
            123: 'NOP',
            132: 'GSW',
            138: 'NOP',
            146: 'GSW',
            166: 'NOP',
            173: 'GSW',
            178: 'NOP',
    """
    shot_times = {}
    for game_7z in game_7z_filenames:
        game_name = game_7z.split(".7z")[0]
        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        except FileNotFoundError:
            continue

        df_events = pd.read_csv(f"{EVENTS_DIR}/{gameid}.csv")
        game_shot_times = {}
        for (row_idx, row) in df_events.iterrows():
            period = row["PERIOD"]
            game_clock = row["PCTIMESTRING"].split(":")
            game_clock_secs = 60 * int(game_clock[0]) + int(game_clock[1])
            game_time = get_game_time(game_clock_secs, period)

            if (row["EVENTMSGTYPE"] == 1) or (row["EVENTMSGTYPE"] == 2):
                game_shot_times[game_time] = row["PLAYER1_TEAM_ABBREVIATION"]

        shot_times[gameid] = game_shot_times
    return shot_times 


### 
# Hoop sides
###


def get_team_hoop_sides(
    game_7z_filenames: List[str],
    shot_times,
) -> Dict[int, Dict[int, Dict[str,int]]]:
    """
    Returns a dict mapping gameid to another dict, which maps
    quarters to an innermost dict, which maps team strings to 
    the x-coordinate of the basket 
    (forced to be either 0 or 94; the COURT_LENGTH is defined to be 94
    in the settings)

    Partial example return value:
        {'0021500295': {1: {'MIL': 94, 'NYK': 0},
                        2: {'MIL': 94, 'NYK': 0},
                        3: {'MIL': 0, 'NYK': 94},
                        4: {'MIL': 0, 'NYK': 94}}}
    """
    hoop_sides = {}
    for game_7z in game_7z_filenames:
        game_name = game_7z.split(".7z")[0]

        try:
            gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        except FileNotFoundError:
            continue

        df_tracking = pd.read_json(f"{TRACKING_DIR}/{game_name}/{gameid}.json")
        hoop_side_counts = {}
        used_game_times = set()
        teams = set()
        for tracking_event in df_tracking["events"]:
            for moment in tracking_event["moments"]:
                period = moment[0]
                game_clock = moment[2]
                game_time = int(get_game_time(game_clock, period))
                if (game_time in shot_times[gameid]) and (
                    game_time not in used_game_times
                ):
                    ball_x = moment[5][0][2]

                    if ball_x < HALF_COURT_LENGTH:
                        hoop_side = 0
                    else:
                        hoop_side = COURT_LENGTH

                    if period not in hoop_side_counts:
                        hoop_side_counts[period] = {}

                    shooting_team = shot_times[gameid][game_time]
                    if shooting_team not in hoop_side_counts[period]:
                        hoop_side_counts[period][shooting_team] = {
                            0: 0,
                            COURT_LENGTH: 0,
                        }

                    hoop_side_counts[period][shooting_team][hoop_side] += 1
                    used_game_times.add(game_time)
                    teams.add(shooting_team)

        if len(teams) == 0:
            print(f"The moments in the {game_name} JSON are empty.", flush=True)
            continue

        hoop_sides[gameid] = get_game_hoop_sides(teams, hoop_side_counts, game_name)
    return hoop_sides 

### 
# Event streams 
###

def _gameid_from_game7z_basename(game_7z_basename: str) -> Optional[str]:
    """

    Argument:
        game_7z_basename: Example '12.17.2015.TOR.at.CHA.7z' or '12.17.2015.TOR.at.CHA'
    
    Returns:
        Example: 0021500492.json
    
    Note:
        The relationship can be determined by inspecting the harddrive, because 
        there are paths of the form
            data/NBA-Player-Movements/<game_7z_basename>/<gameid>.json
        with a single .json per <game_7z_basename>
    """
    # Left strip everything up to the phrase
    game_7z_rstripped=game_7z_basename.rstrip(".7z")

    try:
        gameid = os.listdir(f"{TRACKING_DIR}/{game_7z_rstripped}")[0].split(".")[0]
    except:
        gameid = None 
    
    return gameid 

def get_event_streams(game_7z_filenames: List[str]) -> Tuple[Dict,Dict]:
    """
    First return value (`event2event_idx`):

        This maps event names to event indices:

        {'shot_miss': 0,
        'rebound_defense': 1,
        'layup_block': 2,
        'layup_made': 3,
        'shot_made': 4,
        'rebound_offense': 5,
        'defensive_foul': 6,
        'layup_miss': 7,
        'dunk_made': 8,
        'violation_defense': 9,
        'steal': 10,
        'offensive_foul': 11,
        'timeout': 12,
        'turnover': 13,
        'shot_block': 14,
        'dunk_miss': 15,
        'violation_offense': 16}

    Second return value (`gameid2event_stream`):

        This is a dict whose keys are gameids and whose value is a list of dicts showing
        information about the event (game time, event label, score, etc.). Below is a partial
        example of such a list.

        [{'game_time': 2849,
        'pos_team': 'NOP',
        'event': 'turnover',
        'description': 'holiday out of bounds - bad pass turnover turnover (p4.t16)',
        'event_team': 'NOP',
        'score': '131 - 118'},
        {'game_time': 2867,
        'pos_team': 'GSW',
        'event': 'shot_made',
        'description': "rush 26' 3pt jump shot (5 pts) (speights 2 ast)",
        'event_team': 'GSW',
        'score': '131 - 118'},
        {'game_time': 2877,
        'pos_team': 'NOP',
        'event': 'layup_made',
        'description': "douglas 2' driving layup (17 pts)",
        'event_team': 'NOP',
        'score': '134 - 118'}]

    """

    gameid2event_stream = {}

    for game_7z in game_7z_filenames:
        gameid = _gameid_from_game7z_basename(game_7z) 
        if gameid is not None:
            gameid2event_stream[gameid] = get_event_stream(gameid)

    event2event_idx = {}
    for event_stream in gameid2event_stream.values():
        for event in event_stream:
            event2event_idx.setdefault(event["event"], len(event2event_idx))

    return (event2event_idx, gameid2event_stream)


###
# Save to Disk
###

def save_game_numpy_arrays(
    game_name : str, 
    event_stream_for_one_game : List[Dict],
    hoop_sides: Dict,
    event2event_idx : Dict, 
    playerid2player_idx: Dict[int,int],
    games_dir: str, 
    first_event_idx: Optional[int]=None,
    last_event_idx: Optional[int]=None
) -> None:
    """
    Creates two files, {gameid}_X.npy and {gameid}_y.npy

    The y file is an Numpy1DArray of shape (T,), where T is the number
        of timesteps (or "moments") in the game, where we sample at 25 Hz.
        We expect T to be 25 Hz * 4 quarters * 12 mins * 60 secs/min = 72,000,
        but due to some preprocessing to remove overlapping events, we actually end up
        with slightly less (e.g. '0021500035' has 68,087 events).

        The values in this file are integers, which gives the event description (e.g. 'shot_made', 'layup_block') for
        the event into which the moment is embedded.

    The X file is an np.array of shape (T,D), where D is the dimensionality of "information" about
        each of the T plays.   In particular, we have D=54, and
            0 = elapsed game_time (secs),
            1 = elapsed period_time (secs),
            2 = shot_clock,
            3 = period,
            4 = left_score,  The score of the team whose hoop is on the left.
            5 = right_score, The score of the team whose hoop is on the right.
            6 = left_score - right_score,
            7 = ball_x,
            8 = ball_y,
            9 = ball_z,
            10-19 = the (sorted) player ids of the players involved in the event's play
            20-29 = the player x values, players ordered as above
            30-39 = the player y values, players ordered as above
            40-49 = the player hoop sides, players ordered as above (0=left, 1=right)
            50 = event_id 
            51 = wall_clock
    
    Arguments:
        game_name: E.g. '01.01.2016.CHA.at.TOR.7z'. It's the basename of a folder 
            in data/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs

        event_stream_for_one_game: A List of Dicts. There is one element for each event,
            containining metadata about that event, e.g. the score, and the annotation
            for that event.
        
        hoop_sides: dict, maps gmameid to an inner dict mapping period (1,2,3,4) to where
            each team's hoop was located on the court (either 0 or 94, apparently).  Example:

            In [1]: hoop_sides.keys()
            Out[1]: dict_keys(['0021500492'])

            In [2]: hoop_sides.values()
            Out[2]: dict_values([{1: {'CHA': 94, 'TOR': 0}, 2: {'CHA': 94, 'TOR': 0}, 3: {'CHA': 0, 'TOR': 94}, 4: {'CHA': 0, 'TOR': 94}}])
    
        event2event_idx: dict, maps event labels to event indices, e.g.
            {'offensive_foul': 0,
            'turnover': 1,
            'shot_made': 2,
            'defensive_foul': 3,
            'shot_miss': 4,
        playerid2player_idx : dict, maps player id to player idx, e.g.
            {201949: 0,
            2449: 1,
            201960: 2,
            ....
            1626163: 23,
            201946: 24,
            201150: 25}
    """
    game_name=game_name.rstrip(".7z")

    gameid = _gameid_from_game7z_basename(game_name)

    df_tracking = pd.read_json(f"{TRACKING_DIR}/{game_name}/{gameid}.json")
    home_team = None
    cur_time = -1
    event_idx = 0
    game_over = False
    X = []
    y = []
    start_event_idx = first_event_idx
    stop_event_idx = last_event_idx + 1 if last_event_idx is not None else None 
    for tracking_event in df_tracking["events"][start_event_idx: stop_event_idx]:
        # TODO: To do per-event training, Limit the training for this to the first however many events.
        # Then change the name accordingly. 
        event_id = tracking_event["eventId"]
        if home_team is None:
            home_team_id = tracking_event["home"]["teamid"]
            home_team = TEAM_ID2PROPS[home_team_id]["abbreviation"]

        moments = tracking_event["moments"]
        for moment in moments:
            period = moment[0]
            # Milliseconds.
            wall_clock = moment[1]
            game_clock = moment[2]
            shot_clock = moment[3]
            shot_clock = shot_clock if shot_clock else game_clock

            period_time = 720 - game_clock if period <= 4 else 300 - game_clock
            game_time = get_game_time(game_clock, period)

            # Moments can overlap temporally, so previously processed time points are
            # skipped along with clock stoppages.
            if game_time <= cur_time:
                continue

            while game_time > event_stream_for_one_game[event_idx]["game_time"]:
                event_idx += 1
                if event_idx >= len(event_stream_for_one_game):
                    game_over = True
                    break

            if game_over:
                break

            event = event_stream_for_one_game[event_idx]
            score = event["score"]
            (away_score, home_score) = (int(s) for s in score.split(" - "))
            home_hoop_side = hoop_sides[gameid][period][home_team]
            if home_hoop_side == 0:
                (left_score, right_score) = (home_score, away_score)
            else:
                (right_score, left_score) = (home_score, away_score)

            (ball_x, ball_y, ball_z) = moment[5][0][2:5]
            data = [
                game_time,
                period_time,
                shot_clock,
                period,
                left_score,
                right_score,
                left_score - right_score,
                ball_x,
                ball_y,
                ball_z,
            ]

            if len(moment[5][1:]) != 10:
                continue

            player_idxs = []
            player_xs = []
            player_ys = []
            player_hoop_sides = []

            try:
                for player in moment[5][1:]:
                    player_idxs.append(playerid2player_idx[player[1]])
                    player_xs.append(player[2])
                    player_ys.append(player[3])
                    hoop_side = hoop_sides[gameid][period][
                        TEAM_ID2PROPS[player[0]]["abbreviation"]
                    ]
                    player_hoop_sides.append(int(hoop_side == COURT_LENGTH))

            except KeyError:
                if player[1] == 0:
                    print(
                        f"Bad player in event {event_id} for {game_name}.", flush=True
                    )
                    continue

                else:
                    raise KeyError

            order = np.argsort(player_idxs)
            for idx in order:
                data.append(player_idxs[idx])

            for idx in order:
                data.append(player_xs[idx])

            for idx in order:
                data.append(player_ys[idx])

            for idx in order:
                data.append(player_hoop_sides[idx])

            data.append(event_idx)
            data.append(wall_clock)

            if len(data) != 52:
                raise ValueError

            X.append(np.array(data))
            y.append(event2event_idx.setdefault(event["event"], len(event2event_idx)))
            cur_time = game_time

        if game_over:
            break

    X = np.stack(X)
    y = np.array(y)

    X = add_score_changes(X)

    if (first_event_idx is not None) or (last_event_idx is not None):
        np.save(f"{games_dir}/{gameid}_event_idxs_{first_event_idx}_to_{last_event_idx}__X.npy", X)
        np.save(f"{games_dir}/{gameid}_event_idxs_{first_event_idx}_to_{last_event_idx}__y.npy", y)
    else:
        np.save(f"{games_dir}/{gameid}_X.npy", X)
        np.save(f"{games_dir}/{gameid}_y.npy", y)


def save_numpy_arrays(
    game_7z_filenames: List[str], 
    gameid2event_stream,
    hoop_sides: Dict,
    event2event_idx : Dict, 
    playerid2player_idx: Dict[int,int],
    games_dir: str, 
):

    for game_7z in game_7z_filenames:

        game_name=game_7z.split(".7z")[0]
        # gameid = os.listdir(f"{TRACKING_DIR}/{game_name}")[0].split(".")[0]
        # df_tracking = pd.read_json(f"{TRACKING_DIR}/{game_name}/{gameid}.json")

        try:
            gameid = _gameid_from_game7z_basename(game_name)
            event_stream_for_one_game = gameid2event_stream[gameid]
            save_game_numpy_arrays(game_name, event_stream_for_one_game, hoop_sides, event2event_idx, playerid2player_idx, games_dir)
        except ValueError:
            pass

        # # TODO: Should this be uncommented, as in the original code by Alcorn? If so, why?
        # shutil.rmtree(f"{TRACKING_DIR}/{game_name}")

def save_baller2vec_info(
    player_idx2props: Dict[int, Dict],
    player_idx2playing_time: Dict[int,float],
    event2event_idx: Dict,
    info_dir: str, 
) -> None:

    """
    A `baller2vec_info' simply has two items.  
      - player_idx2props, which maps a player index to {name, playerid, and playing time}:   
          Note that some players may not have played.  An example:
    
          7: {'name': 'Miles Plumlee', 'playerid': 203101},
          8: {'name': 'Rashad Vaughn',
              'playerid': 1626173,
               'playing_time': 191.67299999999213},
    
      - event2event_idx, which maps event labels to event indices, e.g.
          {'offensive_foul': 0,
          'turnover': 1,
          'shot_made': 2,
          'defensive_foul': 3,
          'shot_miss': 4,

    MTW: I always rewrite the config because it don't currently see how it would
    take substantial additional time to do so, and that way the dicts will
    track changes in number of processed data samples. 
    
    try:
        baller2vec_info = pickle.load(
            open(f"{DATA_DIR}/baller2vec_info.pydict", "rb")
        )
        player_idx2props = baller2vec_info["player_idx2props"]
        event2event_idx = baller2vec_info["event2event_idx"]
        playerid2player_idx = {}
        for (player_idx, props) in player_idx2props.items():
            playerid2player_idx[props["playerid"]] = player_idx
    except FileNotFoundError:
        baller2vec_info = False
    """

    for (player_idx, playing_time) in player_idx2playing_time.items():
        player_idx2props[player_idx]["playing_time"] = playing_time

    baller2vec_info = {
        "player_idx2props": player_idx2props,
        "event2event_idx": event2event_idx,
    }
    pickle.dump(
        baller2vec_info, open(f"{info_dir}/baller2vec_info.pydict", "wb")
    )

