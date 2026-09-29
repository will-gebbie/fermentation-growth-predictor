# Data

The reactor data used to develop this project is proprietary and is not included in
the repository. To run the pipeline, place one workbook per reactor run in
`data/formatted/`, named `<RUN_ID>_DATA.xlsx` (e.g. `RUN01_DATA.xlsx`).

## Workbook schema

Each workbook contains three sheets, prefixed with the run ID.

### `<RUN_ID> Bioflo` — process controller log (~1-minute resolution)

| Column        | Description                        |
| ------------- | ---------------------------------- |
| `timestamp`   | Datetime                           |
| `DO`          | Dissolved oxygen (%)               |
| `agitation`   | Agitation (rpm)                    |
| `pH`          | pH                                 |
| `temperature` | Temperature (°C)                   |
| `flo`         | Total inlet gas flow (standard L/min) |

### `<RUN_ID> Gas` — BlueVis gas analyzer (~10-minute resolution)

The analyzer measures the mixed inlet gas and the reactor off-gas on separate ports.
Inlet readings are tared against the outlet sensors using an empty-reactor
measurement, so that in − out is zero when nothing is growing.

| Column      | BlueVis port | Description               |
| ----------- | ------------ | ------------------------- |
| `timestamp` |              | Datetime                  |
| `O2_in`     | O2(3), tared | Inlet O2 (%)              |
| `O2_out`    | O2(2)        | Outlet O2 (%)             |
| `CH4_in`    | CH4(3), tared| Inlet CH4 (%)             |
| `CH4_out`   | CH4(1)       | Outlet CH4 (%)            |
| `CO2_out`   | CO2(1)       | Outlet CO2 (%)            |

### `<RUN_ID> Manual` — operator log

| Column      | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `batch`     | Batch number within the run (increments at each harvest)     |
| `timestamp` | Datetime of the sample / event                               |
| `OD`        | Manually measured optical density (training label; may be blank) |
| `notes`     | Free-text operator notes (gas changes, antifoam, harvests…)  |
