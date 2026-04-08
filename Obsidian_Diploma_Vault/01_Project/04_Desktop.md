# Desktop

Related diagrams:

- [[04_Diagrams/01_System_Architecture_Diagram]]
- [[04_Diagrams/04_Compare_Mode_Diagram]]
- [[04_Diagrams/05_Live_Webcam_Diagram]]

## Technology

The desktop client is built with PySide6.

It demonstrates that the project is a real operator tool, not only a set of API endpoints.

## Main pages

- `Dashboard`
- `Face Search`
- `Database`
- `Logs`

## Dashboard

Dashboard shows:

- backend status;
- available pipelines;
- summary metrics;
- recent activity.

It is the operator landing page.

## Face Search

This is the main working screen.

Workflow:

1. choose an image or start the camera;
2. choose a mode;
3. run the action;
4. inspect result summary;
5. inspect matches and per-face overlays.

Modes:

- `Search`
- `Enroll`
- `Compare`

The page also contains:

- image preview;
- matches table;
- detected faces;
- advanced options;
- technical details.

Live webcam mode keeps preview local and sends selected frames to the backend.
It supports multi-face search and compare flows.

## Database

Database shows person records and details.

It is used for:

- viewing registered records;
- deleting records;
- checking gallery contents;
- verifying enroll/import results.

Delete actions use the admin key path.

## Logs

Logs is the operator maintenance screen.

It is used for:

- backend status;
- activity and audit-oriented visibility;
- index tools;
- rebuild actions;
- general maintenance.
