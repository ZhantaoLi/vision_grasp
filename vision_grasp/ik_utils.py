#!/usr/bin/env python3
"""IK 工具函数 — URDF 解析、正运动学、逆运动学。

被 trajectory_node 和 ik_solver_node 共享。
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def _rpy_to_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle_matrix(axis, angle):
    k = np.array(axis, dtype=float)
    k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _parse_urdf_joints(urdf_path):
    tree = ET.parse(urdf_path)
    joints = {}
    for j in tree.getroot().findall('joint'):
        name = j.get('name')
        origin = j.find('origin')
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get('xyz'):
                xyz = [float(v) for v in origin.get('xyz').split()]
            if origin.get('rpy'):
                rpy = [float(v) for v in origin.get('rpy').split()]
        ax_el = j.find('axis')
        axis = [0.0, 0.0, 1.0]
        if ax_el is not None and ax_el.get('xyz'):
            axis = [float(v) for v in ax_el.get('xyz').split()]
        limit = None
        lim_el = j.find('limit')
        if lim_el is not None:
            limit = {
                'lower': float(lim_el.get('lower', '0')),
                'upper': float(lim_el.get('upper', '0')),
            }
        joints[name] = {
            'parent': j.find('parent').get('link'),
            'child': j.find('child').get('link'),
            'type': j.get('type', 'fixed'),
            'xyz': xyz, 'rpy': rpy, 'axis': axis, 'limit': limit,
        }
    return joints


def _find_chain(joints, base, tip):
    c2j = {v['child']: (k, v) for k, v in joints.items()}
    path, cur = [], tip
    for _ in range(20):
        if cur not in c2j:
            break
        jn, jd = c2j[cur]
        path.append((jn, jd))
        cur = jd['parent']
        if cur == base:
            path.reverse()
            return path
    return None


def _fk(q, chain):
    T = np.eye(4)
    for jd, qi in zip(chain, q):
        To = np.eye(4)
        To[:3, :3] = _rpy_to_matrix(*jd['rpy'])
        To[:3, 3] = jd['xyz']
        Tj = np.eye(4)
        Tj[:3, :3] = _axis_angle_matrix(jd['axis'], qi)
        T = T @ To @ Tj
    return T


def _fk_all_links(q, chain):
    """返回每个关节子 link 在基座系中的 z 坐标列表。"""
    T = np.eye(4)
    zs = []
    for jd, qi in zip(chain, q):
        To = np.eye(4)
        To[:3, :3] = _rpy_to_matrix(*jd['rpy'])
        To[:3, 3] = jd['xyz']
        Tj = np.eye(4)
        Tj[:3, :3] = _axis_angle_matrix(jd['axis'], qi)
        T = T @ To @ Tj
        zs.append(T[2, 3])
    return zs


def _ik_position(target_pos, chain, chain_limits=None, q_init=None,
                 max_iter=500, tol=1e-4):
    n = len(chain)
    q = np.array(q_init if q_init is not None else [0.0] * n)
    for _ in range(max_iter):
        T = _fk(q, chain)
        err = target_pos - T[:3, 3]
        if np.linalg.norm(err) < tol:
            return q, True
        J = np.zeros((3, n))
        delta = 1e-7
        for j in range(n):
            qd = q.copy()
            qd[j] += delta
            Td = _fk(qd, chain)
            J[:, j] = (Td[:3, 3] - T[:3, 3]) / delta
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01**2 * np.eye(3), err)
        q += dq
        if chain_limits is not None:
            lo = np.array([lim[0] for lim in chain_limits])
            hi = np.array([lim[1] for lim in chain_limits])
            q = np.clip(q, lo, hi)
        else:
            q = np.clip(q, -3.14, 3.14)
    T = _fk(q, chain)
    return q, np.linalg.norm(target_pos - T[:3, 3]) < tol * 5
