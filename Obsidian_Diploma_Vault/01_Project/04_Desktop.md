# Desktop

Связанные схемы:

- [[04_Diagrams/01_System_Architecture_Diagram]]
- [[04_Diagrams/04_Compare_Mode_Diagram]]
- [[04_Diagrams/05_Live_Webcam_Diagram]]

## Технология

Desktop-клиент реализован на PySide6.

Он нужен для того, чтобы показать, что система — это не просто набор API endpoints, а операторский инструмент.

## Основные страницы

- `Dashboard`
- `Face Search`
- `Database`
- `Logs`

## Dashboard

Dashboard показывает:

- состояние backend;
- доступные pipeline;
- summary metrics;
- recent activity.

Это стартовая страница оператора.

## Face Search

Это главный рабочий экран.

Workflow:

1. выбрать изображение или запустить камеру;
2. выбрать режим;
3. запустить действие;
4. получить result summary;
5. посмотреть список совпадений.

Поддерживаемые режимы:

- `Search`
- `Enroll`
- `Compare`

Также есть:

- image preview;
- matches table;
- detected faces;
- advanced options;
- technical details.

## Database

Database page показывает записи людей и их детали.

Используется для:

- просмотра person records;
- удаления записей;
- проверки содержимого базы;
- визуального контроля результата enroll/import.

## Logs

Logs page нужен для:

- просмотра состояния backend;
- просмотра событий;
- проверки index tools;
- rebuild индекса;
- maintenance сценариев.
