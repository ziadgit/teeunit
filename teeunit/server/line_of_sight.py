"""
TeeUnit Line of Sight Module

Implements visibility calculations using Bresenham's line algorithm.
"""

from typing import Callable, List, Set, Tuple


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Generate all cells along a line from (x0, y0) to (x1, y1).
    
    Uses Bresenham's line algorithm for efficient integer-only calculation.
    
    Args:
        x0, y0: Starting position
        x1, y1: Ending position
    
    Returns:
        List of (x, y) positions along the line (excluding start, including end)
    """
    cells = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    x, y = x0, y0
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    if dx > dy:
        err = dx / 2
        while x != x1:
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
            cells.append((x, y))
    else:
        err = dy / 2
        while y != y1:
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
            cells.append((x, y))
    
    return cells


def has_line_of_sight(
    x0: int, y0: int,
    x1: int, y1: int,
    is_blocking: Callable[[int, int], bool],
) -> bool:
    """
    Check if there is a clear line of sight between two positions.
    
    Args:
        x0, y0: Observer position
        x1, y1: Target position
        is_blocking: Function that returns True if a cell blocks LOS
    
    Returns:
        True if there is clear line of sight
    """
    # Same cell = visible
    if x0 == x1 and y0 == y1:
        return True
    
    # Get all cells along the line
    line = bresenham_line(x0, y0, x1, y1)
    
    # Check each cell except the last (target)
    for i, (x, y) in enumerate(line):
        if i == len(line) - 1:
            # Don't check the target cell itself
            break
        if is_blocking(x, y):
            return False
    
    return True


def get_visible_cells(
    x: int, y: int,
    radius: int,
    is_blocking: Callable[[int, int], bool],
    in_bounds: Callable[[int, int], bool],
) -> Set[Tuple[int, int]]:
    """
    Get all cells visible from a position within a radius.
    
    Uses raycasting to determine visibility.
    
    Args:
        x, y: Observer position
        radius: Vision radius
        is_blocking: Function that returns True if a cell blocks LOS
        in_bounds: Function that returns True if a cell is within arena bounds
    
    Returns:
        Set of (x, y) positions that are visible
    """
    visible = {(x, y)}  # Observer's cell is always visible
    
    # Cast rays to all cells on the perimeter of the vision square
    # and to cells within the radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            
            target_x = x + dx
            target_y = y + dy
            
            if not in_bounds(target_x, target_y):
                continue
            
            # Check Chebyshev distance (square vision)
            if max(abs(dx), abs(dy)) > radius:
                continue
            
            # Check line of sight
            if has_line_of_sight(x, y, target_x, target_y, is_blocking):
                visible.add((target_x, target_y))
    
    return visible


def get_visible_agents(
    observer_pos: Tuple[int, int],
    agent_positions: dict,  # agent_id -> (x, y)
    observer_id: int,
    radius: int,
    is_blocking: Callable[[int, int], bool],
) -> List[int]:
    """
    Get IDs of agents visible to the observer.
    
    Args:
        observer_pos: Observer's position
        agent_positions: Dict mapping agent_id to position
        observer_id: Observer's agent ID (excluded from results)
        radius: Vision radius
        is_blocking: Function that returns True if a cell blocks LOS
    
    Returns:
        List of visible agent IDs
    """
    visible_agents = []
    ox, oy = observer_pos
    
    for agent_id, (ax, ay) in agent_positions.items():
        if agent_id == observer_id:
            continue
        
        # Check distance
        dx = abs(ax - ox)
        dy = abs(ay - oy)
        
        if max(dx, dy) > radius:
            continue
        
        # Check line of sight
        if has_line_of_sight(ox, oy, ax, ay, is_blocking):
            visible_agents.append(agent_id)
    
    return visible_agents


def get_cells_along_shot(
    x0: int, y0: int,
    x1: int, y1: int,
    max_range: int,
    is_blocking: Callable[[int, int], bool],
) -> List[Tuple[int, int]]:
    """
    Get all cells a shot passes through, stopping at walls or max range.
    
    Args:
        x0, y0: Shooter position
        x1, y1: Target position
        max_range: Maximum shot range
        is_blocking: Function that returns True if a cell blocks shots
    
    Returns:
        List of (x, y) positions the shot passes through
    """
    # Extend the line to max_range if target is beyond it
    dx = x1 - x0
    dy = y1 - y0
    
    if dx == 0 and dy == 0:
        return []
    
    # Normalize direction and extend to max_range
    import math
    length = math.sqrt(dx * dx + dy * dy)
    
    if length > max_range:
        # Clip to max range
        scale = max_range / length
        x1 = x0 + int(dx * scale)
        y1 = y0 + int(dy * scale)
    
    # Get cells along the line
    line = bresenham_line(x0, y0, x1, y1)
    
    # Stop at blocking cells
    result = []
    for x, y in line:
        result.append((x, y))
        if is_blocking(x, y):
            break
    
    return result


def cast_ray(
    x0: int, y0: int,
    dx: int, dy: int,
    max_distance: int,
    is_blocking: Callable[[int, int], bool],
    in_bounds: Callable[[int, int], bool],
) -> Tuple[int, int]:
    """
    Cast a ray in a direction and find where it stops.
    
    Args:
        x0, y0: Starting position
        dx, dy: Direction (will be normalized to unit steps)
        max_distance: Maximum ray distance
        is_blocking: Function that returns True if a cell blocks the ray
        in_bounds: Function that returns True if a cell is within bounds
    
    Returns:
        (x, y) position where the ray stopped
    """
    if dx == 0 and dy == 0:
        return (x0, y0)
    
    # Calculate end point at max distance
    import math
    length = math.sqrt(dx * dx + dy * dy)
    scale = max_distance / length
    
    x1 = x0 + int(dx * scale)
    y1 = y0 + int(dy * scale)
    
    # Get cells along the ray
    line = bresenham_line(x0, y0, x1, y1)
    
    # Find where it stops
    for x, y in line:
        if not in_bounds(x, y):
            return (x0, y0) if not line else line[line.index((x, y)) - 1] if line.index((x, y)) > 0 else (x0, y0)
        if is_blocking(x, y):
            return (x, y)
    
    return (x1, y1) if line else (x0, y0)


def get_sound_direction(
    listener_pos: Tuple[int, int],
    source_pos: Tuple[int, int],
    max_hearing_distance: int = 10,
) -> str:
    """
    Get approximate direction of a sound source.
    
    Args:
        listener_pos: Position of the listener
        source_pos: Position of the sound source
        max_hearing_distance: Maximum distance sounds can be heard
    
    Returns:
        Direction string, or empty string if too far
    """
    dx = source_pos[0] - listener_pos[0]
    dy = source_pos[1] - listener_pos[1]
    
    import math
    distance = math.sqrt(dx * dx + dy * dy)
    
    if distance > max_hearing_distance:
        return ""
    
    # Get general direction
    if abs(dx) < 2 and abs(dy) < 2:
        return "nearby"
    
    # Primary direction
    h_dir = "east" if dx > 0 else "west" if dx < 0 else ""
    v_dir = "south" if dy > 0 else "north" if dy < 0 else ""
    
    # Determine primary direction based on angle
    if abs(dx) > abs(dy) * 2:
        return h_dir
    elif abs(dy) > abs(dx) * 2:
        return v_dir
    elif h_dir and v_dir:
        return f"{v_dir}{h_dir}"
    else:
        return h_dir or v_dir or "nearby"
