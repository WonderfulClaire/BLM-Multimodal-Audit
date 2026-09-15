"""Independent parser for numeric-fixture-v1 observations, not a telecom expert.

Only the explicitly versioned synthetic grammar is accepted. Missing measurements
are rejected rather than silently interpreted as normal. Metadata labels are ignored.
"""
import re

RULE_VERSION = 'numeric-fixture-v1'


def validate_measured_case(case):
    sections = case['sections']
    counts = {'mobility': 1, 'antenna': 1, 'cell_relation': 3, 'handover': 2, 'resource': 1}
    for view, count in counts.items():
        records = sections.get(view)
        if not isinstance(records, list) or len(records) != count:
            raise ValueError(f'Unrecognized or ambiguous records: {view}')

    def extract(view, index, pattern):
        record = sections[view][index]
        match = re.fullmatch(pattern, record) if isinstance(record, str) else None
        if match is None: raise ValueError(f'Unrecognized measurement: {view}/{index}')
        return match.groups()

    speed, = extract('mobility', 0, r'Test vehicle speed: (\d+) km/h\. Maximum permitted speed: 40 km/h\.')
    tilt, coverage = extract('antenna', 0, r'Serving antenna downtilt: (\d+) degrees\. (Excessive tilt weakens far-end coverage\.|Far-end coverage is adequate\.)')
    distance, rsrp = extract('cell_relation', 0, r'Serving-cell distance: (\d+(?:\.\d+)?) km; serving RSRP: (-?\d+) dBm\.')
    carrier, = extract('cell_relation', 1, r'The non-colocated neighbor (uses the same carrier and causes severe interference\.|uses a different carrier; interference is absent\.)')
    pci, neighbor = extract('cell_relation', 2, r'Serving PCI: (\d+); neighbor PCI: (\d+)\.')
    events, pingpong = extract('handover', 0, r'Handover events per minute: (\d+)\. (Repeated ping-pong returns to the previous cell\.|No ping-pong events\.)')
    threshold, description = extract('handover', 1, r'A3 threshold: (\d+) dB\. (The threshold is misconfigured and blocks transfer to a stronger neighbor\.|The threshold is correctly configured\.)')
    blocks, = extract('resource', 0, r'Average scheduled resource blocks: (\d+); required minimum: 160\.')
    if (int(threshold) > 10) != ('misconfigured' in description):
        raise ValueError('Contradictory threshold description')
    flags = [int(speed) > 40, int(tilt) > 12 and coverage.startswith('Excessive'),
             float(distance) > 1 and int(rsrp) < -105, carrier.startswith('uses the same'),
             int(pci) % 30 == int(neighbor) % 30,
             int(events) >= 10 and pingpong.startswith('Repeated'), int(threshold) > 10,
             int(blocks) < 160]
    return [f'C{i+1}' for i, flag in enumerate(flags) if flag]
