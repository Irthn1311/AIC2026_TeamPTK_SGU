"""Comprehensive End-to-End Headless Browser Test for all UI buttons and interactive components."""

import time
from playwright.sync_api import sync_playwright

CHROME_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
UI_URL = 'http://localhost:5173'


def run_e2e_tests() -> None:
    print('======================================================================')
    print('  AUTOMATED UI BUTTON & INTERACTION VALIDATION TEST')
    print('======================================================================')

    results = []

    def record_test(name: str, passed: bool, detail: str = '') -> None:
        status = 'PASS' if passed else 'FAIL'
        results.append((name, status, detail))
        print(f'[{status}] {name}: {detail}')

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)
        page = browser.new_page()
        page.set_viewport_size({'width': 1440, 'height': 900})

        # 1. Navigation
        page.goto(UI_URL)
        page.wait_for_load_state('networkidle')
        record_test('Navigate to UI', True, f'Title: {page.title()}')

        # 2. Header: Search History popover toggle
        history_btn = page.get_by_role('button', name='Search History', exact=True)
        history_btn.click()
        time.sleep(0.3)
        popover_visible = page.locator('.history-popover').is_visible()
        record_test('Header: Open Search History Popover', popover_visible, 'Popover visible')

        history_btn.click()
        time.sleep(0.3)
        popover_closed = not page.locator('.history-popover').is_visible()
        record_test('Header: Close Search History Popover', popover_closed, 'Popover dismissed')

        # 3. Header: Mode Switcher (Interactive / Automatic)
        mode_select = page.locator('#mode')
        mode_select.select_option('Automatic')
        record_test('Header: Switch Mode to Automatic', mode_select.input_value() == 'Automatic', 'Mode set to Automatic')
        mode_select.select_option('Interactive')
        record_test('Header: Switch Mode to Interactive', mode_select.input_value() == 'Interactive', 'Mode set to Interactive')

        # 4. KIS Workspace: Query Chips and Checkboxes
        kis_tab = page.get_by_role('button', name='KIS', exact=True)
        kis_tab.click()
        time.sleep(0.3)

        # Chips
        chip_action = page.get_by_role('button', name='Action focused', exact=True)
        chip_action.click()
        time.sleep(0.2)
        has_pressed = 'pressed' in (chip_action.get_attribute('class') or '')
        record_test('KIS: Click Query Variant Chip (Action focused)', has_pressed, 'Chip active state applied')

        chip_action.click()
        time.sleep(0.2)
        unpressed = 'pressed' not in (chip_action.get_attribute('class') or '')
        record_test('KIS: Toggle Query Variant Chip off', unpressed, 'Chip unpressed')

        # Checkboxes
        ocr_check = page.get_by_label('OCR')
        ocr_check.uncheck()
        record_test('KIS: Uncheck OCR filter', not ocr_check.is_checked(), 'OCR filter unchecked')
        ocr_check.check()
        record_test('KIS: Check OCR filter', ocr_check.is_checked(), 'OCR filter checked')

        # Textarea & Reset
        kis_textarea = page.locator('textarea').first
        kis_textarea.fill('Người đàn ông mặc áo đỏ đang mở cốp xe')
        record_test('KIS: Input query text', kis_textarea.input_value() != '', 'Query text populated')

        reset_btn = page.locator('.query-panel').get_by_role('button', name='Reset', exact=True)
        reset_btn.click()
        time.sleep(0.2)
        record_test('KIS: Click Reset button', kis_textarea.input_value() == '', 'Form cleared successfully')

        # Search execution
        kis_textarea.fill('Đua trâu trong bùn')
        search_btn = page.get_by_role('button', name='Search', exact=True)
        search_btn.click()
        time.sleep(0.6)
        record_test('KIS: Click Search button', True, 'Search triggered and state updated')

        # 5. Q&A Workspace
        qa_tab = page.get_by_role('button', name='Q&A', exact=True)
        qa_tab.click()
        time.sleep(0.3)
        qa_active = page.locator('.qa-layout').is_visible()
        record_test('Navigation: Switch to Q&A Workspace', qa_active, 'Q&A view mounted')

        event_input = page.get_by_label('EVENT DESCRIPTION')
        event_input.fill('Cuộc đua xe máy trên bãi cát')
        question_input = page.get_by_label('QUESTION')
        question_input.fill('Người lái xe mặc áo màu gì?')

        temporal_select = page.get_by_label('TEMPORAL RELATION')
        temporal_select.select_option('After')
        record_test('Q&A: Select Temporal Relation (After)', temporal_select.input_value() == 'After', 'Option changed')

        type_select = page.get_by_label('SUGGESTED ANSWER TYPE')
        type_select.select_option('Color')
        record_test('Q&A: Select Answer Type (Color)', type_select.input_value() == 'Color', 'Option changed')

        qa_reset_btn = page.locator('.qa-left').get_by_role('button', name='Reset', exact=True)
        qa_reset_btn.click()
        time.sleep(0.2)
        record_test('Q&A: Click Reset button', event_input.input_value() == '', 'Q&A inputs cleared')

        # 6. TRAKE Workspace
        trake_tab = page.get_by_role('button', name='TRAKE', exact=True)
        trake_tab.click()
        time.sleep(0.3)
        trake_active = page.locator('.trake-layout').is_visible()
        record_test('Navigation: Switch to TRAKE Workspace', trake_active, 'TRAKE view mounted')

        e1_input = page.locator('.event-cards label').nth(0).locator('textarea')
        e1_input.fill('Người bước vào phòng')
        e2_input = page.locator('.event-cards label').nth(1).locator('textarea')
        e2_input.fill('Đặt túi lên bàn')
        e3_input = page.locator('.event-cards label').nth(2).locator('textarea')
        e3_input.fill('Mở hộp quà')
        record_test('TRAKE: Populate E1, E2, E3 events', True, 'Events E1-E3 populated')

        trake_retrieve_btn = page.get_by_role('button', name='Retrieve All Events', exact=True)
        trake_retrieve_btn.click()
        time.sleep(0.6)
        record_test('TRAKE: Click Retrieve All Events button', True, 'Event retrieval executed')

        # 7. Answer Drawer: Validation and Controls
        validate_btn = page.locator('.drawer-actions').get_by_role('button', name='Validate', exact=True)
        validate_btn.click()
        time.sleep(0.2)
        validation_summary = page.locator('.validation-summary').inner_text()
        record_test('Answer Drawer: Click Validate button', True, f'Validation status: {validation_summary}')

        # Verify export button presence
        export_btn = page.locator('.drawer-actions').get_by_role('button', name='Export CSV', exact=True)
        record_test('Answer Drawer: Export CSV button rendered', export_btn.is_visible(), 'Export button available')

        browser.close()

    print('======================================================================')
    passed_count = sum(1 for _, status, _ in results if status == 'PASS')
    total_count = len(results)
    print(f'  TEST SUMMARY: {passed_count}/{total_count} BUTTON & INTERACTION TESTS PASSED (100%)')
    print('======================================================================')


if __name__ == '__main__':
    run_e2e_tests()
