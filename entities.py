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

    @staticmethod
    def from_dict(json_dict: Dict) -> 'KeyEvent':
        key = str(json_dict.get("Key"))
        event = KeyboardEventType(json_dict.get("Event"))
        input_box = InputBox(json_dict.get("Input Box"))
        text_changed = bool(json_dict.get("Text Changed"))
        timestamp = datetime.strptime(json_dict.get("Timestamp"), '%Y-%m-%d %H:%M:%S.%f')
        epoch = float(json_dict.get("Epoch"))
        return KeyboardEvent(key, event, input_box, text_changed, timestamp, epoch)


@dataclass
class Coordinates:
    x: float
    y: float

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

    @staticmethod
    def from_dict(json_dict: Dict) -> 'MouseEvent':
        event = MouseEventType(json_dict.get("Event"))
        coordinates = Coordinates.from_dict(json_dict.get("Coordinates"))
        timestamp = datetime.strptime(json_dict.get("Timestamp"), '%Y-%m-%d %H:%M:%S.%f')
        epoch = float(json_dict.get("Epoch"))
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
        for x in json_list:
            if "false_enters" in x:
                false_enters = int(x.get("false_enters"))
            else:
                events.append(MouseEvent.from_dict(x))
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
        keyboard_events = [KeyboardEvent.from_dict(x) for x in json_dict.get("key_events")]
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
        return User(user_id, details, test_cases)