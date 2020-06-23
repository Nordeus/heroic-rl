import re

import yaml

from ..train.enums import Rarity, Spell, Unit

SPELL_PATTERN = re.compile("^SPELL_.+$")
UNIT_PATTERN = re.compile("^UNIT_.+$")
RARITY_PATTERN = re.compile("^RARITY_.+$")


def spell_yaml_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", "SPELL_" + data.name)


def spell_yaml_constructor(loader, node):
    value = loader.construct_scalar(node)
    # Skip 'SPELL_' prefix
    return Spell[value[6:]]


def unit_yaml_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", "UNIT_" + data.name)


def unit_yaml_constructor(loader, node):
    value = loader.construct_scalar(node)
    # Skip 'UNIT_' prefix
    return Unit[value[5:]]


def rarity_yaml_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", "RARITY_" + data.name)


def rarity_yaml_constructor(loader, node):
    value = loader.construct_scalar(node)
    # Skip 'RARITY_' prefix
    return Rarity[value[7:]]


def safe_load_yaml(yaml_string):
    yaml.add_implicit_resolver("!Spell", SPELL_PATTERN, Loader=yaml.SafeLoader)
    yaml.add_implicit_resolver("!Unit", UNIT_PATTERN, Loader=yaml.SafeLoader)
    yaml.add_implicit_resolver("!Rarity", RARITY_PATTERN, Loader=yaml.SafeLoader)
    yaml.add_constructor("!Spell", spell_yaml_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!Unit", unit_yaml_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!Rarity", rarity_yaml_constructor, Loader=yaml.SafeLoader)
    return yaml.safe_load(yaml_string)


def safe_dump_yaml(obj):
    yaml.SafeDumper.ignore_aliases = lambda *args: True
    yaml.add_representer(Spell, spell_yaml_representer, Dumper=yaml.SafeDumper)
    yaml.add_representer(Unit, unit_yaml_representer, Dumper=yaml.SafeDumper)
    yaml.add_representer(Rarity, rarity_yaml_representer, Dumper=yaml.SafeDumper)
    return yaml.safe_dump(obj, explicit_start=True)
