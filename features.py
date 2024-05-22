from dataclasses import dataclass

import pandas as pd

from entities import KeyboardEvent, MouseEvents, TestCase, MouseEvent, Coordinates, User
from typing import List, Tuple, Dict


@dataclass
class PressReleaseEventGroup:
    press: KeyboardEvent | MouseEvent
    release: KeyboardEvent | MouseEvent
    time_pressed: float

    def __init__(self, press: KeyboardEvent | MouseEvent, release: KeyboardEvent | MouseEvent):
        self.press = press
        self.release = release
        self.time_pressed = self.get_time_pressed()

    def get_time_pressed(self) -> float:
        if self.press is None or self.release is None:
            return 0
        return self.release.epoch - self.press.epoch


@dataclass
class ReleasePressEventGroup:
    release: KeyboardEvent | MouseEvent
    press: KeyboardEvent | MouseEvent
    time_not_pressed: float

    def __init__(self, release: KeyboardEvent | MouseEvent, press: KeyboardEvent | MouseEvent):
        self.release = release
        self.press = press
        self.time_not_pressed = self.get_time_not_pressed()

    def get_time_not_pressed(self) -> float:
        if self.press is None or self.release is None:
            return 0
        return self.press.epoch - self.release.epoch


@dataclass
class MovementEventGroup:
    end: MouseEvent
    start: MouseEvent
    time_between_moves: float

    def __init__(self, end: MouseEvent, start: MouseEvent):
        self.start = start
        self.end = end
        self.time_between_moves = self.get_time_between_moves()

    def get_time_between_moves(self) -> float:
        if self.start is None or self.end is None:
            return 0
        return self.start.epoch - self.end.epoch


@dataclass
class MovementTrajectoryEventGroup:
    events: List[MouseEvent]
    time_moved: float
    distance_moved: float

    def __init__(self, events: List[MouseEvent]):
        self.events = events
        self.time_moved = self.get_time_moved()
        self.distance_moved = self.get_distance_moved()

    def get_time_moved(self) -> float:
        return self.events[-1].epoch - self.events[0].epoch

    def get_distance_moved(self) -> float:
        distance = 0
        previous_event = self.events[0]
        for event in self.events[1:]:
            distance += Coordinates.distance(previous_event.coordinates, event.coordinates)
            previous_event = event
        return distance


@dataclass
class TestFeatures:
    test_case: TestCase
    keyboard_press_release: List[PressReleaseEventGroup]
    keyboard_release_press: List[ReleasePressEventGroup]
    mouse_press_release: List[PressReleaseEventGroup]
    mouse_release_press: List[ReleasePressEventGroup]
    mouse_movement: List[MovementEventGroup]
    mouse_trajectory: List[MovementTrajectoryEventGroup]


@dataclass
class UserFeatures:
    user: User
    test_features: List[TestFeatures]


def group_keyboard_events(events: List[KeyboardEvent]) -> (
        Tuple)[List[PressReleaseEventGroup], List[ReleasePressEventGroup]]:
    press_release_event_groups = []
    release_press_event_groups = []
    for index, keyboard_event in enumerate(events):
        if keyboard_event.is_pressed():
            for keyboard_sub_event in events[index:]:
                if keyboard_event.is_pair(keyboard_sub_event):
                    press_release_event_groups.append(PressReleaseEventGroup(
                        keyboard_event,
                        keyboard_sub_event)
                    )
                    break
        if keyboard_event.is_released():
            for keyboard_sub_event in events[index:]:
                if keyboard_sub_event.is_pressed():
                    release_press_event_groups.append(ReleasePressEventGroup(
                        keyboard_event,
                        keyboard_sub_event)
                    )
                    break
    press_release_event_groups.sort(key=lambda x: x.press.epoch)
    release_press_event_groups.sort(key=lambda x: x.release.epoch)
    return press_release_event_groups, release_press_event_groups


def group_mouse_events(event: MouseEvents) -> (
        Tuple)[
    List[PressReleaseEventGroup],
    List[ReleasePressEventGroup],
    List[MovementTrajectoryEventGroup],
    List[MovementEventGroup]
]:
    press_release_event_groups = []
    release_press_event_groups = []
    movement_trajectory_event_groups = []
    movement_event_groups = []
    movement_ids = []
    movement_index = None

    for index, mouse_event in enumerate(event.events):
        if mouse_event.is_press():
            for mouse_sub_event in event.events[index + 1:]:
                if mouse_event.is_pair(mouse_sub_event):
                    press_release_event_groups.append(PressReleaseEventGroup(
                        mouse_event,
                        mouse_sub_event)
                    )
                    break
        if mouse_event.is_release():
            for mouse_sub_event in event.events[index + 1:]:
                if mouse_sub_event.is_press():
                    release_press_event_groups.append(ReleasePressEventGroup(
                        mouse_event,
                        mouse_sub_event)
                    )
                    break
        if mouse_event.is_movement() and mouse_event.movement_id not in movement_ids:
            if movement_index is None:
                movement_index = index + 1
            movements = [mouse_event]
            for sub_index, mouse_sub_event in enumerate(event.events[movement_index:]):
                if mouse_sub_event.is_movement():
                    if mouse_event.is_same_movement(mouse_sub_event):
                        movements.append(mouse_sub_event)
                    else:
                        movement_event_groups.append(MovementEventGroup(
                            movements[-1],
                            mouse_sub_event)
                        )
                        movement_trajectory_event_groups.append(MovementTrajectoryEventGroup(
                            movements)
                        )
                        movement_index += sub_index + 1
                        movement_ids.append(mouse_event.movement_id)
                        break

    return (press_release_event_groups,
            release_press_event_groups,
            movement_trajectory_event_groups,
            movement_event_groups)


def test_case_features(case: TestCase) -> TestFeatures:
    keyboard_press_release, keyboard_release_press = group_keyboard_events(case.keyboard_events)
    mouse_press_release, mouse_release_press, mouse_trajectory, mouse_movement = group_mouse_events(case.mouse_events)
    return TestFeatures(
        case,
        keyboard_press_release,
        keyboard_release_press,
        mouse_press_release,
        mouse_release_press,
        mouse_movement,
        mouse_trajectory
    )


def user_features(user: User) -> UserFeatures:
    return UserFeatures(
        user,
        [test_case_features(test_case) for test_case in user.test_cases]
    )


def generate_features(users: List[User]) -> List[UserFeatures]:
    return [user_features(user) for user in users]


def convert_feature_to_dataframe(features: List[UserFeatures]) -> Dict[int, pd.DataFrame]:
    user_features_dict = {}

    for user_feature in features:
        features_dataframe = pd.DataFrame(columns=['keyboard_dwell', 'keyboard_flight',
                                                   'mouse_trajectory_distance', 'mouse_trajectory_time', 'mouse_dwell',
                                                   'mouse_flight', 'valid'])
        for test_feature in user_feature.test_features:
            max_len = max(len(test_feature.keyboard_press_release), len(test_feature.keyboard_release_press),
                          len(test_feature.mouse_trajectory), len(test_feature.mouse_movement),
                          len(test_feature.mouse_press_release), len(test_feature.mouse_release_press))
            for i in range(max_len):
                new_dataframe = pd.DataFrame(
                    [{
                        'keyboard_dwell': test_feature.keyboard_press_release[i].time_pressed if i < len(
                            test_feature.keyboard_press_release) else 0.0,
                        'keyboard_flight': test_feature.keyboard_release_press[i].time_not_pressed if i < len(
                            test_feature.keyboard_release_press) else 0.0,
                        'mouse_trajectory_distance': test_feature.mouse_trajectory[i].distance_moved if i < len(
                            test_feature.mouse_trajectory) else 0.0,
                        'mouse_trajectory_time': test_feature.mouse_trajectory[i].time_moved if i < len(
                            test_feature.mouse_trajectory) else 0.0,
                        'mouse_dwell': test_feature.mouse_press_release[i].time_pressed if i < len(
                            test_feature.mouse_press_release) else 0.0,
                        'mouse_flight': test_feature.mouse_release_press[i].time_not_pressed if i < len(
                            test_feature.mouse_release_press) else 0.0,
                        'valid': 1 if test_feature.test_case.positive else 0
                    }]
                )
                # print(new_dataframe.to_string(index=False))
                features_dataframe = pd.concat([features_dataframe, new_dataframe], ignore_index=True)
        user_features_dict[user_feature.user.identifier] = features_dataframe.query(
            'keyboard_dwell != 0 or '
            'keyboard_flight != 0 or '
            'mouse_trajectory_distance != 0 or '
            'mouse_trajectory_time != 0 or '
            'mouse_dwell != 0 or '
            'mouse_flight != 0')
    return user_features_dict
