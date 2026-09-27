# INF1771

## Running

```bash
python main.py                 # console output
python main.py --visual        # opens the pygame window and runs everything live
python -m unittest discover    # 39 tests
```

`pip install pygame` is only needed for the window. Without it `--visual` and
`visualize.py` fail with an explicit message and the console run is unaffected.

## Live visualization

`python visualize.py` (or `python main.py --visual`) opens a window where the
search is drawn while it runs. The pipeline executes in a worker thread and the
window renders at 60 FPS, so the interface never freezes.

Phases, in order:

1. **Local A\* on the grid** - all 325 legs, one at a time, with the expanded set,
   the frontier and the agent visible in real time.
2. **Global A\*** - the 24-gym search does not live on the grid, so the window
   shows it as live counters (expanded states, queue size, best cost) and draws
   the best order found so far as a polyline. It keeps the same
   `--route-budget` wall-clock limit as the console run, so the animation never
   changes the result.
3. **Replay of the final path** - the agent walks the 458-cell route slowly and
   the HUD accumulates the movement cost per terrain type.
4. **Genetic algorithm** - the band below the map shows the team chosen in each
   of the 24 battles (`05 T  Weedle, Caterpie  66.7`) and the remaining energy of
   the five Pokémon; the right-hand side panel keeps the run summary
   and the cost-per-generation chart. The number next to the
   team is the **battle time** (`gym difficulty / sum of the team power`), which is
   exactly what the GA minimizes. Every color is documented: `green = best of the
   generation`, `gray = population average`, and in the energy block
   `filled square = what is left, empty = what was already spent`. `T` hides the
   band. The whole history of the 120 generations of the 30 runs is recorded, so
   `G` replays it later; `--ga-animation` animates it live.

Colors: `.` white, `M` brown, `F` green, `A` blue, `R` gray. Every special cell is
magenta at any zoom level, so a gym never blends into the terrain; zoomed in past
16 px the start gets a white ring and the destination an orange one, and below that
the start is a dark square and the destination an orange one. Overlays are drawn
on top: expanded states purple, frontier yellow, current leg path cyan, best global
order orange, agent red.

The window opens showing the **whole map**, so the complete route from `01` to
`26` is always on screen. Every start/gym/destination square is colored in every
zoom level; the `01`-`26` badges need at least a 16 px cell, so they only appear
once you zoom in. Because a 150-column map cannot be shown at 16 px on one screen,
the camera then follows the agent (or the cell being expanded) and pans only when
it approaches an edge, so the current gym is always visible. Dragging with the
mouse turns the follow off; `C` turns it back on.

The cyan line means different things per phase, and the layer legend and the
cost block are labelled accordingly: during the local A* phase it is one of the
325 legs (`current leg`, `leg`) and only touches the two gyms of that leg,
while during the replay it is the final route (`final path`), which always
starts at `01`, ends at `26`, and touches every gym exactly once. The swatch of
that row in `cost per terrain` is cyan in both phases, because the line it
accounts for is the cyan one. The `cell X/Y` progress bar is cyan while the
agent walks and turns yellow only when the walk is finished.

The agent always walks on its own at 22 cells per second; the HUD shows
`cell X/Y` with a progress bar so it is obvious that it is moving, and says
whether it is running or paused. Nothing has to be clicked to make it move.

| Key | Action |
| --- | --- |
| `SPACE` | pause / resume |
| `+` `-` | animation speed (0.05x to unlimited) |
| `F` | nudge the agent forward a few cells, then keep running |
| `TAB` | skip the current phase |
| `V` `B` `P` | toggle expanded / frontier / path |
| `L` | cycle layer presets |
| `G` | replay the genetic algorithm history |
| `T` | show / hide the genetic algorithm band (under the map) |
| `Home` | toggle whole-map view / readable zoom with follow |
| `C` | recenter on the agent and re-enable follow |
| `R` / `ESC` | restart / quit |
| wheel | zoom the map |
| drag | pan the map |

## Remaining work

- [ ] Make the exact global A* search finish within a reasonable time for all 24 gyms on the official map.
- [x] Complete the terminal visualization: show the agent, final path, frontier, visited states, and accumulated movement cost on every terrain type.
- [ ] Add an integration test that runs the complete workflow with `mapa.txt` and validates the gym order, route cost, battle cost, total cost, final energy, and expanded-state counts.
- [x] Make the test suite discoverable with the default `python -m unittest discover` command.
- [ ] Run and record the final 30-run genetic-algorithm experiment after the route search is fixed.
- [ ] Document how A* and the genetic algorithm work, the experiment results, and the limits of the optimality guarantee.
- [ ] Prepare the required presentation video with every group member participating.
- [ ] Add and commit the implementation, map, and tests before submission.
