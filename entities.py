import math
from datetime import datetime
from enum import Enum
from typing import List, Dict
from dataclasses import dataclass


class Month(Enum):
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12


class KeyboardEventType(Enum):
    PRESSED = 'pressed'
    RELEASED = 'released'


class MouseEventType(Enum):
    MOVEMENT = 'movement'
    LEFT_PRESS = 'left press'
    LEFT_RELEASE = 'left release'
    RIGHT_PRESS = 'right press'
    RIGHT_RELEASE = 'right release'
    SCROLL_UP_PRESS = 'scrollup press'
    SCROLL_UP_RELEASE = 'scrollup release'
    SCROLL_DOWN_PRESS = 'scrolldown press'
    SCROLL_DOWN_RELEASE = 'scrolldown release'
    SCROLL_RIGHT_PRESS = 'scrollright press'
    SCROLL_RIGHT_RELEASE = 'scrollright release'
    SCROLL_LEFT_PRESS = 'scrollleft press'
    SCROLL_LEFT_RELEASE = 'scrollleft release'
    MOUSE_4_PRESS = 'mouse4 press'
    MOUSE_4_RELEASE = 'mouse4 release'


keyword_event_type_reverse = {
    KeyboardEventType.PRESSED: KeyboardEventType.RELEASED
}

mouse_event_type_reverse = {
    MouseEventType.LEFT_PRESS: MouseEventType.LEFT_RELEASE,
    MouseEventType.RIGHT_PRESS: MouseEventType.RIGHT_RELEASE,
    MouseEventType.SCROLL_UP_PRESS: MouseEventType.SCROLL_UP_RELEASE,
    MouseEventType.SCROLL_DOWN_PRESS: MouseEventType.SCROLL_DOWN_RELEASE,
    MouseEventType.SCROLL_RIGHT_PRESS: MouseEventType.SCROLL_RIGHT_RELEASE,
    MouseEventType.SCROLL_LEFT_PRESS: MouseEventType.SCROLL_LEFT_RELEASE,
    MouseEventType.MOUSE_4_PRESS: MouseEventType.MOUSE_4_RELEASE
}

mouse_event_type_press = {
    MouseEventType.LEFT_PRESS,
    MouseEventType.RIGHT_PRESS,
    MouseEventType.SCROLL_UP_PRESS,
    MouseEventType.SCROLL_DOWN_PRESS,
    MouseEventType.SCROLL_RIGHT_PRESS,
    MouseEventType.SCROLL_LEFT_PRESS,
    MouseEventType.MOUSE_4_PRESS
}

mouse_event_type_release = {
    MouseEventType.LEFT_RELEASE,
    MouseEventType.RIGHT_RELEASE,
    MouseEventType.SCROLL_UP_RELEASE,
    MouseEventType.SCROLL_DOWN_RELEASE,
    MouseEventType.SCROLL_RIGHT_RELEASE,
    MouseEventType.SCROLL_LEFT_RELEASE,
    MouseEventType.MOUSE_4_RELEASE
}


class InputBox(Enum):
    NONE = 'Null'
    NAME = 'Name'
    CARD_NUMBER = 'Card No'
    CVC = 'CVC'
    EXPIRY_MONTH = 'Exp m'
    EXPIRY_YEAR = 'Exp y'


@dataclass
class Details:
    identifier: str
    provider: str
    name: str
    card_number: float
    cvc: int
    expiry_month: Month
    expiry_year: int

    @staticmethod
    def from_dict(json_dict: Dict) -> 'Details':
        identifier = str(json_dict.get("ID"))
        provider = str(json_dict.get("Provider"))
        name = str(json_dict.get("Name"))
        card_number = float(json_dict.get("Card Number"))
        cvc = int(json_dict.get("CVC"))
        expiry = str(json_dict.get("Expiry"))
        expiry_month = Month(int(expiry.split('/')[0]))
        expiry_year = int('20' + expiry.split('/')[1])
        return Details(identifier, provider, name, card_number, cvc, expiry_month, expiry_year)


@dataclass
class KeyboardEvent:
    key: str
    event: KeyboardEventType
    input_box: InputBox
    text_changed: bool
    timestamp: datetime
    epoch: float

    def is_pair(self, other: 'KeyboardEvent') -> bool:
        return self.key == other.key and other.event == keyword_event_type_reverse.get(self.event)

    def is_pressed(self) -> bool:
        return self.event == KeyboardEventType.PRESSED

    def is_released(self) -> bool:
        return self.event == KeyboardEventType.RELEASED

    @staticmethod
    def from_dict(json_dict: Dict) -> 'KeyboardEvent':
        key = str(json_dict.get("Key"))
        event = KeyboardEventType(json_dict.get("Event"))
        input_box = InputBox(json_dict.get("Input Box"))
        text_changed = bool(json_dict.get("Text Changed"))
        timestamp = datetime.strptime(json_dict.get("Timestamp"), '%Y-%m-%d %H:%M:%S.%f')
        epoch = float(json_dict.get("Epoch")) * 1000
        return KeyboardEvent(key, event, input_box, text_changed, timestamp, epoch)


@dataclass
class Coordinates:
    x: float
    y: float

    @staticmethod
    def distance(cord0: 'Coordinates', cord1: 'Coordinates'):
        return math.sqrt((cord1.x - cord0.x) ** 2 + (cord1.y - cord0.y) ** 2)

    @staticmethod
    def from_dict(coordinates: List[float]) -> 'Coordinates':
        x = float(coordinates[0])
        y = float(coordinates[1])
        return Coordinates(x, y)


@dataclass
class MouseEvent:
    event: MouseEventType
    coordinates: Coordinates
    timestamp: datetime
    epoch: float
    movement_id: int | None

    def is_pair(self, other: 'MouseEvent') -> bool:
        return other.event == mouse_event_type_reverse.get(self.event)

    def is_press(self) -> bool:
        return self.event in mouse_event_type_press

    def is_release(self) -> bool:
        return self.event in mouse_event_type_release

    def is_movement(self) -> bool:
        return self.event == MouseEventType.MOVEMENT

    def is_same_movement(self, other: 'MouseEvent') -> bool:
        if self.movement_id is None:
            return False
        return self.movement_id == other.movement_id

    @staticmethod
    def from_dict(json_dict: Dict) -> 'MouseEvent':
        event = MouseEventType(json_dict.get("Event"))
        coordinates = Coordinates.from_dict(json_dict.get("Coordinates"))
        timestamp = datetime.strptime(json_dict.get("Timestamp"), '%Y-%m-%d %H:%M:%S.%f')
        epoch = float(json_dict.get("Epoch")) * 1000
        movement_id = int(json_dict.get("Movement ID")) if "Movement ID" in json_dict else None
        return MouseEvent(event, coordinates, timestamp, epoch, movement_id)


@dataclass
class MouseEvents:
    events: List[MouseEvent]
    false_enters: int | None

    @staticmethod
    def from_dict(json_list: List[Dict]) -> 'MouseEvents':
        false_enters = None
        events = []
        for event in json_list:
            if "false_enters" in event:
                false_enters = int(event.get("false_enters"))
            else:
                events.append(MouseEvent.from_dict(event))
        events.sort(key=lambda x: x.epoch)
        return MouseEvents(events, false_enters)


@dataclass
class TestCase:
    identifier: int
    positive: bool
    keyboard_events: List[KeyboardEvent]
    mouse_events: MouseEvents

    @staticmethod
    def from_dict(json_dict: Dict, key: str, positive: bool) -> 'TestCase':
        identifier = int(key.split('_')[1])
        keyboard_events = list(KeyboardEvent.from_dict(x) for x in json_dict.get("key_events"))
        keyboard_events.sort(key=lambda x: x.epoch)
        mouse_events = MouseEvents.from_dict(json_dict.get("mouse_events"))
        return TestCase(identifier, positive, keyboard_events, mouse_events)


@dataclass
class User:
    identifier: int
    details: Details
    test_cases: List[TestCase]

    @staticmethod
    def from_dict(obj: Dict, filename: str) -> 'User':
        user_id = int(filename.split('_')[-1].split('.')[0])
        details = Details.from_dict(obj.get("details"))
        test_cases = []
        for key, value in obj.get("true_data").items():
            test_cases.append(TestCase.from_dict(value, key, True))
        for key, value in obj.get("false_data").items():
            test_cases.append(TestCase.from_dict(value, key, False))
        test_cases.sort(key=lambda x: (not x.positive, x.identifier))
        return User(user_id, details, test_cases)
