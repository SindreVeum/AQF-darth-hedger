"""Recover Figure 7's plotted nodes from the journal PDF's vector paths.

Run: python -m src.extract_paper_figure7 [--pdf PATH]
Requires pypdf for this optional extraction step. Pricing reads the saved JSON
and does not need pypdf. These are digitized plotting coordinates, not the
author's original numerical output or evidence of 50 independent PDE solves.
"""

import argparse
import csv
import hashlib
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream

from src.hagan import hagan_iv


def rounding_maps(ticks, coordinates, half_unit=.005):
    """Affine calibrations consistent with rounded axis coordinates.

    The half-unit covers coordinates printed to two decimals. Bounds apply only
    to PDF coordinate rounding; they do not bound the author's pricing error.
    """
    vertices = []
    for i, j in combinations(range(len(ticks)), 2):
        for si, sj in product((-1, 1), repeat=2):
            a, b = np.linalg.solve([[ticks[i], 1], [ticks[j], 1]],
                                  [coordinates[i] + si*half_unit,
                                   coordinates[j] + sj*half_unit])
            if a > 0 and np.max(abs(a*ticks+b-coordinates)) <= half_unit+1e-9:
                vertices.append((a, b))
    if not vertices:
        raise ValueError('Axis coordinates do not admit the assumed rounding bound.')
    return np.asarray(vertices)


def coordinate_bounds(coordinates, maps, half_unit=.005):
    a, b = maps.T
    low = ((coordinates[:, None]-half_unit-b)/a).min(axis=1)
    high = ((coordinates[:, None]+half_unit-b)/a).max(axis=1)
    return low, high


def extract(pdf):
    reader = PdfReader(pdf)
    form = reader.pages[22]['/Resources']['/XObject']['/Im8'].get_object()
    paths, path, color = [], [], None
    for operands, op in ContentStream(form, reader).operations:
        if op == b'K':
            color = [float(x) for x in operands]
        elif op in (b'm', b'l'):
            path.append([float(x) for x in operands])
        elif op in (b'S', b's', b'f', b'f*', b'B', b'n'):
            if op == b'S':
                paths.append((np.asarray(path), color))
            path = []
    curves = [(p, c) for p, c in paths if p.shape == (50, 2)]
    assert len(curves) == 4
    assert all(np.array_equal(p[:, 0], curves[0][0][:, 0]) for p, _ in curves)
    # Legend order and CMYK colours were visually verified on the complete page.
    names = ['ann', 'hagan', 'ln', 'fdm']
    assert np.allclose(curves[3][1], [.101563, .972656, .980469, .015625])
    xt = np.array(sorted(p[0, 0] for p, _ in paths if p.shape == (2, 2)
                         and p[0, 0] == p[1, 0] and p[0, 1] == 124.691
                         and p[1, 1] < p[0, 1]))
    yt = np.array(sorted(p[0, 1] for p, _ in paths if p.shape == (2, 2)
                         and p[0, 1] == p[1, 1] and p[0, 0] == 234.594
                         and p[1, 0] == 219.602))
    kt = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5])
    vt = np.arange(30, 71, 5)/100
    assert len(xt) == len(kt) and len(yt) == len(vt)
    ax, bx = np.polyfit(np.log10(kt), xt, 1)
    ay, by = np.polyfit(vt, yt, 1)
    K = 10**((curves[0][0][:, 0]-bx)/ax)
    klo, khi = coordinate_bounds(curves[0][0][:, 0], rounding_maps(np.log10(kt), xt))
    data = dict(K=K.tolist(), K_lower=(10**klo).tolist(), K_upper=(10**khi).tolist())
    bounds, residual = {}, {}
    for name, (p, _) in zip(names, curves):
        values = (p[:, 1]-by)/ay
        data[f'paper_{name}_iv'] = values.tolist()
        lo, hi = coordinate_bounds(p[:, 1], rounding_maps(vt, yt))
        bounds[name] = {'lower': lo.tolist(), 'upper': hi.tolist(),
                        'max_coordinate_rounding_iv_bp':
                            float(max(np.max(values-lo), np.max(hi-values))*1e4)}
    for days in (365, 360):
        err = (hagan_iv(1., K, 540/days, .3, 1., -.35, .6)
               - np.asarray(data['paper_hagan_iv']))*1e4
        residual[str(days)] = {'mae_iv_bp': float(np.mean(abs(err))),
                              'max_abs_iv_bp': float(np.max(abs(err)))}
    assert residual['365']['max_abs_iv_bp'] < .05
    assert residual['360']['max_abs_iv_bp'] > 1.
    return {
        'source_pdf': str(pdf.resolve()),
        'source_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'source_page_1based': 23, 'source_form': '/Im8', 'figure': 7,
        'reference_kind': 'digitized_pdf_vector_nodes_not_author_raw_output',
        'normalization': {'F': 1., 'beta': 1.},
        'parameters': {'T': 540/365, 'alpha': .3, 'nu': .6, 'rho': -.35},
        'nodes': 50, **data,
        'mapping': {
            'x_definition': 'PDF_x = ax * log10(K) + bx', 'ax': ax, 'bx': bx,
            'y_definition': 'PDF_y = ay * decimal_IV + by', 'ay': ay, 'by': by,
            'x_tick_strikes': kt.tolist(), 'x_tick_coordinates': xt.tolist(),
            'y_tick_decimal_iv': vt.tolist(), 'y_tick_coordinates': yt.tolist(),
            'x_max_tick_residual': float(max(abs(ax*np.log10(kt)+bx-xt))),
            'y_max_tick_residual': float(max(abs(ay*vt+by-yt))),
            'curve_order': names, 'curve_cmyk': [c for _, c in curves],
            'raw_curve_coordinates': {n: p.tolist() for n, (p, _) in zip(names, curves)},
        },
        'coordinate_rounding': {
            'assumed_max_half_unit': .005,
            'meaning': 'Feasible affine maps with all tick coordinates and each curve coordinate within +/-0.005 source units. These bounds exclude author pricing, interpolation and other publication errors.',
            'iv_bounds': bounds,
        },
        'day_count_check': residual,
        'limitations': [
            'The caption gives 540 days but no day-count basis. 365 is supported by the independently computed Hagan curve, not fitted to FDM.',
            'The 50 PDF vertices share strike positions. The author does not specify whether each was priced independently or obtained from interpolation.',
            'K is recovered from logarithmic axis ticks; no guessed exact linspace or training-strike formula replaces these digitized positions.',
            'PDF coordinate precision is not an error bound on the published FDM scheme.',
        ],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, default=root.parent/'McGhee 2020 ANN SABR.pdf')
    parser.add_argument('--output', type=Path, default=root/'results/fdm_paper_direct')
    args = parser.parse_args()
    result = extract(args.pdf)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'figure7_reference.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    names = ['K', 'K_lower', 'K_upper', 'paper_ann_iv', 'paper_hagan_iv', 'paper_ln_iv', 'paper_fdm_iv']
    with (args.output/'figure7_reference.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(zip(*(result[name] for name in names)))
    print(json.dumps({'nodes': result['nodes'], 'K_range': [result['K'][0], result['K'][-1]],
                      'day_count_check': result['day_count_check'],
                      'max_fdm_coordinate_rounding_iv_bp': result['coordinate_rounding']['iv_bounds']['fdm']['max_coordinate_rounding_iv_bp']}, indent=2))


if __name__ == '__main__':
    main()
