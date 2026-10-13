import asyncio
import contextlib
import io
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from core.action_loader import validate_tool_arguments
from core import confirm
from core import llm_client


class ConfigTests(unittest.TestCase):
    def test_environment_key_takes_precedence_and_is_not_persisted(self):
        from memory import config_manager

        with tempfile.TemporaryDirectory() as temp:
            config_file = Path(temp) / "api_keys.json"
            config_file.write_text('{"assistant_name":"JARVIS"}', encoding="utf-8")
            with (
                mock.patch.object(config_manager, "CONFIG_FILE", config_file),
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "environment-key"}),
            ):
                self.assertEqual(config_manager.get_gemini_key(), "environment-key")
                self.assertNotIn("environment-key", config_file.read_text(encoding="utf-8"))

    def test_invalid_config_is_not_overwritten_by_a_save(self):
        from memory import config_manager

        with tempfile.TemporaryDirectory() as temp:
            config_dir = Path(temp)
            config_file = config_dir / "api_keys.json"
            invalid_contents = "{not json"
            config_file.write_text(invalid_contents, encoding="utf-8")
            with (
                mock.patch.object(config_manager, "CONFIG_DIR", config_dir),
                mock.patch.object(config_manager, "CONFIG_FILE", config_file),
            ):
                with self.assertRaisesRegex(RuntimeError, "unreadable or invalid JSON"):
                    config_manager.save_assistant_config("JARVIS", "user")
            self.assertEqual(config_file.read_text(encoding="utf-8"), invalid_contents)


class ToolArgumentTests(unittest.TestCase):
    def test_rejects_non_object_and_missing_required_values(self):
        schema = {
            "type": "OBJECT",
            "properties": {"path": {"type": "STRING"}},
            "required": ["path"],
        }
        self.assertEqual(validate_tool_arguments([], schema),
                         "arguments must be an object")
        self.assertEqual(validate_tool_arguments({}, schema),
                         "arguments.path is required")

    def test_rejects_wrong_types_and_extra_properties(self):
        schema = {
            "type": "OBJECT",
            "properties": {"count": {"type": "INTEGER"}},
            "additionalProperties": False,
        }
        self.assertEqual(validate_tool_arguments({"count": True}, schema),
                         "arguments.count must be integer")
        self.assertIn("is not allowed",
                      validate_tool_arguments({"other": 1}, schema))


class ShutdownTests(unittest.TestCase):
    def tearDown(self):
        confirm.bind(None, None)

    def test_shutdown_tool_waits_for_human_confirmation(self):
        from main import JarvisLive

        class UI:
            muted = False

            def set_state(self, _state):
                pass

        class Registry:
            @staticmethod
            def scheduling(_name):
                return None

        shown = []
        confirm.bind(lambda title, detail: shown.append((title, detail)),
                     lambda: None)
        jarvis = JarvisLive.__new__(JarvisLive)
        jarvis.ui = UI()
        jarvis._loop = None
        jarvis._action_registry = Registry()
        jarvis._plugin_registry = Registry()
        response = asyncio.run(jarvis._execute_tool(types.SimpleNamespace(
            name="shutdown_jarvis", args={}, id="shutdown-test"
        )))
        self.assertIn("CONFIRMATION_PENDING", response.response["result"])
        self.assertEqual(shown[0][0], "Shut down JARVIS")
        self.assertEqual(confirm.pending_title(), "Shut down JARVIS")
        confirm.resolve(False)

    def test_graceful_shutdown_saves_session_and_stops_waiters(self):
        from main import JarvisLive, _ShutdownSignal
        from unittest.mock import AsyncMock

        class UI:
            def write_log(self, _message):
                pass

        async def exercise():
            jarvis = JarvisLive.__new__(JarvisLive)
            jarvis.ui = UI()
            jarvis._shutdown_event = asyncio.Event()
            jarvis._shutdown_started = False
            jarvis._save_session_summary = AsyncMock()
            jarvis.session = types.SimpleNamespace(
                send_client_content=AsyncMock()
            )
            with mock.patch("main.asyncio.sleep", new=AsyncMock()):
                await jarvis._graceful_shutdown()
            jarvis._save_session_summary.assert_awaited_once()
            jarvis.session.send_client_content.assert_awaited_once()
            self.assertTrue(jarvis._shutdown_event.is_set())
            with self.assertRaises(_ShutdownSignal):
                await jarvis._wait_for_shutdown()

        asyncio.run(exercise())


class PluginTests(unittest.TestCase):
    def test_plugin_crash_is_reported_to_caller_and_logger(self):
        from core import plugin_loader

        def fail(_parameters):
            raise RuntimeError("expected test failure")

        record = plugin_loader.PluginRecord(
            name="crashing_plugin", run=fail, valid=True
        )
        messages = []
        registry = plugin_loader.PluginRegistry(
            {"crashing_plugin": record}, messages.append
        )
        with (
            mock.patch.object(plugin_loader, "get_plugin_enabled", return_value=True),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = registry.run("crashing_plugin", {})
        self.assertIn("plugin failed", result)
        self.assertTrue(any("crashed during run()" in message for message in messages))

    def test_plugin_arguments_are_validated_before_execution(self):
        from core import plugin_loader

        called = []
        record = plugin_loader.PluginRecord(
            name="validated_plugin",
            parameters={
                "type": "OBJECT",
                "properties": {"path": {"type": "STRING"}},
                "required": ["path"],
            },
            run=lambda parameters: called.append(parameters) or "done",
            valid=True,
        )
        registry = plugin_loader.PluginRegistry({"validated_plugin": record},
                                                lambda _message: None)
        with mock.patch.object(plugin_loader, "get_plugin_enabled", return_value=True):
            result = registry.run("validated_plugin", {"path": 3})
        self.assertIn("Invalid arguments", result)
        self.assertEqual(called, [])


class LLMClientTests(unittest.TestCase):
    def test_missing_gemini_key_fails_before_client_creation(self):
        from core import gemini

        with mock.patch.object(gemini, "api_key", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "no Gemini API key"):
                gemini.client()

    def test_unavailable_ollama_is_reported_without_waiting(self):
        with (
            mock.patch.object(llm_client, "get_llm_provider", return_value="ollama"),
            mock.patch.object(llm_client, "get_llm_settings",
                              return_value=("http://localhost:11434", "missing-model")),
            mock.patch.object(llm_client.requests, "get",
                              side_effect=llm_client.requests.ConnectionError),
            mock.patch.object(llm_client.subprocess, "Popen",
                              side_effect=FileNotFoundError),
        ):
            self.assertFalse(llm_client.ensure_ollama_running(timeout=0))

    def test_text_request_uses_openai_compatible_backend(self):
        response = mock.Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "  local answer  "}}]
        }
        response.raise_for_status.return_value = None
        with (
            mock.patch.object(llm_client, "get_llm_settings",
                              return_value=("http://localhost:1234", "chosen-model")),
            mock.patch.object(llm_client, "get_llm_provider", return_value="openai"),
            mock.patch.object(llm_client.requests, "post", return_value=response) as post,
        ):
            self.assertEqual(llm_client.call_llm_text("prompt"), "local answer")
        self.assertEqual(post.call_args.args[0],
                         "http://localhost:1234/v1/chat/completions")

    def test_openai_timeout_is_reported_without_ollama_retry(self):
        with (
            mock.patch.object(llm_client, "get_llm_settings",
                              return_value=("http://localhost:1234", "chosen-model")),
            mock.patch.object(llm_client, "get_llm_provider", return_value="openai"),
            mock.patch.object(llm_client.requests, "post",
                              side_effect=llm_client.requests.Timeout),
            mock.patch.object(llm_client, "ensure_ollama_running") as ensure,
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                llm_client.call_llm_text("prompt", timeout=2)
        ensure.assert_not_called()

    def test_bad_provider_type_falls_back_to_ollama(self):
        with mock.patch.object(llm_client, "_load_config",
                               return_value={"llm_provider": None}):
            self.assertEqual(llm_client.get_llm_provider(), "ollama")


class ConfirmationTests(unittest.TestCase):
    def tearDown(self):
        confirm.bind(None, None)

    def test_headless_confirmation_refuses_to_run(self):
        ran = []
        confirm.bind(None, None)
        result = confirm.request("test", "Dangerous action", "details",
                                 lambda: ran.append(True) or "done")
        self.assertIn("interface is not available", result)
        self.assertEqual(ran, [])

    def test_only_one_pending_confirmation_can_be_queued(self):
        shown = []
        ran = []
        confirm.bind(lambda title, detail: shown.append(title), lambda: None)
        first = confirm.request("first", "First action", "details",
                                lambda: ran.append("first") or "done")
        second = confirm.request("second", "Second action", "details",
                                 lambda: ran.append("second") or "done")
        self.assertIn("CONFIRMATION_PENDING", first)
        self.assertIn("already awaiting confirmation", second)
        self.assertEqual(shown, ["First action"])
        confirm.resolve(False)
        self.assertEqual(ran, [])

    def test_delete_waits_for_user_confirmation(self):
        from actions import file_controller

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            target = home / "delete-me.txt"
            target.write_text("keep until confirmed", encoding="utf-8")
            shown = []
            confirm.bind(lambda title, detail: shown.append(title), lambda: None)
            with (
                mock.patch.object(file_controller, "_SAFE_ROOTS", [home]),
                mock.patch.object(file_controller, "_get_desktop", return_value=home / "Desktop"),
                mock.patch.object(file_controller, "_get_downloads", return_value=home / "Downloads"),
                mock.patch.object(file_controller, "_get_documents", return_value=home / "Documents"),
                mock.patch.object(file_controller, "_get_pictures", return_value=home / "Pictures"),
                mock.patch.object(file_controller, "_get_music", return_value=home / "Music"),
                mock.patch.object(file_controller, "_get_videos", return_value=home / "Videos"),
                mock.patch.object(file_controller, "_safe_trash", return_value="Moved to Trash"),
            ):
                result = file_controller.delete_file(str(home), target.name)
                self.assertIn("CONFIRMATION_PENDING", result)
                self.assertTrue(target.exists())
                self.assertEqual(len(shown), 1)
                confirm.resolve(False)
                self.assertTrue(target.exists())


class FileSafetyTests(unittest.TestCase):
    def tearDown(self):
        confirm.bind(None, None)

    def test_game_updates_require_confirmation_and_app_ids_are_validated(self):
        from actions import game_updater

        confirm.bind(lambda *_: None, lambda: None)
        with mock.patch.object(game_updater, "_find_steam_path") as find_steam:
            pending = game_updater.game_updater({
                "action": "install",
                "platform": "steam",
                "game_name": "Example",
            })
            self.assertIn("CONFIRMATION_PENDING", pending)
            find_steam.assert_not_called()
            confirm.resolve(False)

        invalid = game_updater.game_updater({
            "action": "install",
            "platform": "steam",
            "app_id": "123/../../evil",
        })
        self.assertIn("only 1 to 10 digits", invalid)

    def test_sending_a_message_requires_confirmation(self):
        from actions import send_message

        shown = []
        confirm.bind(lambda title, detail: shown.append((title, detail)), lambda: None)
        parameters = {
            "receiver": "Alex",
            "message_text": "Please call me",
            "platform": "WhatsApp",
        }
        with (
            mock.patch.object(send_message, "_PYAUTOGUI", True),
            mock.patch.object(send_message, "_send_whatsapp") as send,
        ):
            result = send_message.send_message(parameters)
            self.assertIn("CONFIRMATION_PENDING", result)
            send.assert_not_called()
            self.assertIn("Please call me", shown[0][1])
            confirm.resolve(False)
            send.assert_not_called()

    def test_live_browser_interactions_require_confirmation(self):
        from actions import browser_control

        shown = []
        confirm.bind(lambda title, detail: shown.append((title, detail)),
                     lambda: None)
        with mock.patch.object(
            browser_control._registry, "get",
            side_effect=AssertionError("browser must not be contacted"),
        ):
            result = browser_control.browser_control({
                "action": "type",
                "text": "example input",
            })
        self.assertIn("CONFIRMATION_PENDING", result)
        self.assertIn("example input", shown[0][1])
        confirm.resolve(False)

    def test_browser_screenshot_paths_stay_in_home_and_do_not_overwrite(self):
        from actions import browser_control

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            desktop = home / "Desktop"
            desktop.mkdir()
            existing = desktop / "jarvis_screenshot.png"
            existing.write_bytes(b"existing")
            with mock.patch.object(browser_control.Path, "home", return_value=home):
                result = browser_control._screenshot_output_path(None)
                self.assertEqual(result, desktop / "jarvis_screenshot_1.png")
                with self.assertRaisesRegex(ValueError, "inside your home folder"):
                    browser_control._screenshot_output_path(
                        str(home.parent / "outside.png")
                    )
            self.assertEqual(existing.read_bytes(), b"existing")

    def test_fill_form_fields_are_declared_in_browser_tool_schema(self):
        from actions import browser_control

        schema = browser_control.TOOL["parameters"]
        self.assertEqual(schema["properties"]["fields"]["type"], "OBJECT")
        self.assertIsNone(validate_tool_arguments(
            {"action": "fill_form", "fields": {"email": "person@example.test"}},
            schema,
        ))

    def test_rename_rejects_path_traversal(self):
        from actions import file_controller

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            folder = home / "folder"
            folder.mkdir()
            source = folder / "source.txt"
            source.write_text("safe", encoding="utf-8")
            with mock.patch.object(file_controller, "_SAFE_ROOTS", [home]):
                result = file_controller.rename_file(str(folder), "source.txt",
                                                     "..\\..\\outside.txt")
            self.assertIn("single file or folder name", result)
            self.assertTrue(source.exists())
            self.assertFalse((home.parent / "outside.txt").exists())

    def test_file_processor_rejects_unselected_local_file(self):
        from actions import file_processor

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "private.txt"
            path.write_text("private data", encoding="utf-8")
            with mock.patch.object(file_processor, "_gemini_client",
                                   side_effect=AssertionError("must not call AI")):
                result = file_processor.file_processor({
                    "file_path": str(path),
                    "action": "summarize",
                })
        self.assertIn("Access denied", result)

    def test_file_processor_requires_confirmation_before_running_selected_code(self):
        from actions import file_processor

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selected.py"
            path.write_text("print('should not run before confirmation')", encoding="utf-8")
            shown = []
            confirm.bind(lambda title, detail: shown.append(title), lambda: None)
            player = mock.Mock(current_file=str(path))
            with mock.patch.object(file_processor.subprocess, "run") as run:
                result = file_processor.file_processor(
                    {"file_path": str(path), "action": "run"}, player=player
                )
                self.assertIn("CONFIRMATION_PENDING", result)
                run.assert_not_called()
                confirm.resolve(False)
                self.assertEqual(len(shown), 1)
                run.assert_not_called()

    def test_xml_is_parsed_as_xml_and_output_names_are_confined(self):
        from actions import file_processor

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "document.xml"
            source.write_text("<root><item>safe</item></root>", encoding="utf-8")
            result = file_processor._process_xml(source, "validate", {})
            self.assertIn("Valid XML", result)
            output = file_processor._output_path(source, "../escape", "../../outside")
            self.assertEqual(output.parent, source.parent)
            self.assertNotIn("..", output.name)

    def test_archive_extraction_rejects_traversal_entries(self):
        import zipfile
        from actions import file_processor

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "must not escape")
            destination = root / "extract"
            result = file_processor._process_archive(
                archive, "extract", {"destination": str(destination)}
            )
            self.assertIn("unsafe path", result)
            self.assertFalse((root / "outside.txt").exists())
            self.assertFalse(destination.exists())

    def test_generated_desktop_code_is_never_executed(self):
        from actions.desktop import desktop_control

        marker = Path(tempfile.gettempdir()) / "jarvis-regression-marker.txt"
        marker.unlink(missing_ok=True)
        result = desktop_control({"action": "task", "task": "do a thing"})
        self.assertIn("disabled", result)
        self.assertFalse(marker.exists())

    def test_code_helper_rejects_unselected_files_before_model_call(self):
        from actions import code_helper

        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp) / "private.py"
            private.write_text("secret = True", encoding="utf-8")
            with mock.patch.object(code_helper, "_get_gemini",
                                   side_effect=AssertionError("must not call AI")):
                result = code_helper.code_helper({
                    "action": "explain",
                    "file_path": str(private),
                })
        self.assertIn("Access denied", result)

    def test_code_execution_waits_for_user_confirmation(self):
        from actions import code_helper

        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / "selected.py"
            script.write_text("print('not run')", encoding="utf-8")
            player = mock.Mock(current_file=str(script))
            shown = []
            confirm.bind(lambda title, detail: shown.append(title), lambda: None)
            with mock.patch.object(code_helper.subprocess, "run") as run:
                result = code_helper.code_helper({
                    "action": "run",
                    "file_path": str(script),
                }, player=player)
                self.assertIn("CONFIRMATION_PENDING", result)
                run.assert_not_called()
                confirm.resolve(False)
                self.assertEqual(len(shown), 1)
                run.assert_not_called()

    def test_generated_code_build_waits_before_model_or_execution(self):
        from actions import code_helper

        confirm.bind(lambda *_: None, lambda: None)
        with (
            mock.patch.object(code_helper, "_get_gemini",
                              side_effect=AssertionError("must not call AI")),
            mock.patch.object(code_helper.subprocess, "run") as run,
        ):
            result = code_helper.code_helper({
                "action": "build",
                "description": "write a harmless script",
            })
        self.assertIn("CONFIRMATION_PENDING", result)
        run.assert_not_called()
        confirm.resolve(False)

    def test_code_helper_cannot_write_outside_home(self):
        from actions import code_helper

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "home"
            home.mkdir()
            outside = Path(temp) / "outside.py"
            with mock.patch.object(Path, "home", return_value=home):
                result = code_helper.code_helper({
                    "action": "write",
                    "description": "write a harmless script",
                    "output_path": str(outside),
                })
        self.assertIn("Invalid output path", result)

    def test_overwrite_waits_for_confirmation_and_preserves_contents(self):
        from actions import code_helper

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            target = home / "existing.py"
            target.write_text("keep this", encoding="utf-8")
            confirm.bind(lambda *_: None, lambda: None)
            with (
                mock.patch.object(Path, "home", return_value=home),
                mock.patch.object(code_helper, "DESKTOP", home),
                mock.patch.object(code_helper, "_get_gemini",
                                  side_effect=AssertionError("must not call AI")),
            ):
                result = code_helper.code_helper({
                    "action": "write",
                    "description": "write a harmless script",
                    "output_path": str(target),
                })
                self.assertIn("CONFIRMATION_PENDING", result)
                confirm.resolve(False)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep this")

    def test_windows_app_launch_never_uses_shell(self):
        from actions import open_app

        with (
            mock.patch.object(open_app.shutil, "which", return_value="C:\\Windows\\app.exe"),
            mock.patch.object(open_app.subprocess, "Popen") as popen,
        ):
            self.assertTrue(open_app._launch_windows("C:\\Windows\\app.exe"))
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["C:\\Windows\\app.exe"])
        self.assertNotIn("shell", kwargs)


class DashboardSecurityTests(unittest.TestCase):
    def test_device_login_rejects_non_object_body_and_rate_limits(self):
        from fastapi.testclient import TestClient
        from dashboard.server import DashboardServer

        client = TestClient(DashboardServer().app)
        self.assertEqual(
            client.post("/api/device-login", json=[]).status_code, 400
        )
        for _ in range(11):
            response = client.post(
                "/api/device-login", json={"device_token": "invalid"}
            )
            self.assertEqual(response.status_code, 401)
        limited = client.post(
            "/api/device-login", json={"device_token": "invalid"}
        )
        self.assertEqual(limited.status_code, 429)

    def test_command_rejects_malformed_json_shape(self):
        from fastapi.testclient import TestClient
        from dashboard.server import DashboardServer

        server = DashboardServer()
        token = "valid-command-token"
        server._tokens.add(token)
        response = TestClient(server.app).post(
            "/api/command", json=[],
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 400)

    def test_logout_revokes_current_session_and_remembered_device(self):
        from fastapi.testclient import TestClient
        from dashboard.server import DashboardServer

        server = DashboardServer()
        token = "active-session-token"
        device_token = "remembered-device-token"
        session_key = "ABC234"
        server._tokens.add(token)
        server._token_keys[token] = session_key
        server._device_sessions[device_token] = {"session_key": session_key}
        server._aes_key(session_key)
        response = TestClient(server.app).post(
            "/api/logout", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(token, server._tokens)
        self.assertNotIn(token, server._token_keys)
        self.assertNotIn(device_token, server._device_sessions)
        self.assertNotIn(session_key, server._aes_cache)

    def test_upload_requires_authentication_and_never_overwrites(self):
        from fastapi.testclient import TestClient
        from dashboard import server as dashboard_server

        if not dashboard_server._UPLOAD_OK:
            self.skipTest("python-multipart is not installed")

        server = dashboard_server.DashboardServer()
        with tempfile.TemporaryDirectory() as temp:
            server._uploads_dir = Path(temp)
            client = TestClient(server.app)
            unauthorized = client.post(
                "/api/upload", files={"file": ("secret.txt", b"content")}
            )
            self.assertEqual(unauthorized.status_code, 401)

            token = "test-upload-token"
            server._tokens.add(token)
            headers = {"Authorization": f"Bearer {token}"}
            first = client.post(
                "/api/upload",
                files={"file": ("..\\private.txt", b"first")},
                headers=headers,
            )
            second = client.post(
                "/api/upload",
                files={"file": ("..\\private.txt", b"second")},
                headers=headers,
            )
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(first.json()["name"], "private.txt")
            self.assertEqual(second.json()["name"], "private_1.txt")
            self.assertEqual((Path(temp) / "private.txt").read_bytes(), b"first")
            self.assertEqual((Path(temp) / "private_1.txt").read_bytes(), b"second")

    def test_upload_size_limit_removes_partial_file(self):
        from fastapi.testclient import TestClient
        from dashboard import server as dashboard_server

        if not dashboard_server._UPLOAD_OK:
            self.skipTest("python-multipart is not installed")

        server = dashboard_server.DashboardServer()
        token = "test-upload-token"
        server._tokens.add(token)
        with tempfile.TemporaryDirectory() as temp:
            server._uploads_dir = Path(temp)
            with mock.patch.object(dashboard_server, "MAX_UPLOAD_MB", 0):
                response = TestClient(server.app).post(
                    "/api/upload",
                    files={"file": ("large.bin", b"too large")},
                    headers={"Authorization": f"Bearer {token}"},
                )
            self.assertEqual(response.status_code, 413)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_download_requires_bearer_auth(self):
        from fastapi.testclient import TestClient
        from dashboard.server import DashboardServer

        server = DashboardServer()
        with tempfile.TemporaryDirectory() as temp:
            server._uploads_dir = Path(temp)
            response = TestClient(server.app).get("/uploads/private.txt")
        self.assertEqual(response.status_code, 401)

    def test_qr_pairing_key_is_posted_not_accepted_from_url(self):
        from fastapi.testclient import TestClient
        from dashboard.server import DashboardServer

        server = DashboardServer()
        key = server.new_key()
        response = TestClient(server.app).post("/api/auto-login", json={"key": key})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertNotIn(key, str(response.url))
        self.assertNotIn(key, server._pending_keys)
        legacy = TestClient(server.app).get(f"/auto-login?key={key}")
        self.assertEqual(legacy.status_code, 404)

    def test_dashboard_binds_loopback_by_default(self):
        from dashboard.server import DashboardServer

        with mock.patch.dict(os.environ, {}, clear=True):
            instance = DashboardServer()
        self.assertFalse(instance.remote_enabled())
        self.assertEqual(instance._host, "127.0.0.1")

    def test_websocket_query_token_does_not_authenticate(self):
        from fastapi.testclient import TestClient
        from dashboard.server import DashboardServer

        server = DashboardServer()
        server._tokens.add("query-token")
        client = TestClient(server.app)
        with client.websocket_connect("/ws?token=query-token") as websocket:
            websocket.send_json({"token": "invalid"})
            with self.assertRaises(Exception):
                websocket.receive_json()

    def test_remote_mode_refuses_plaintext_if_tls_unavailable(self):
        from dashboard import server as dashboard_server

        async def check():
            with mock.patch.dict(os.environ, {"JARVIS_DASHBOARD_HOST": "0.0.0.0"}):
                instance = dashboard_server.DashboardServer()
            with (
                mock.patch.object(dashboard_server, "_ensure_certs", return_value=False),
                mock.patch.object(instance, "_ssl_enabled", return_value=False),
            ):
                with self.assertRaisesRegex(RuntimeError, "requires a usable TLS"):
                    await instance.serve()

        asyncio.run(check())

    def test_certificate_generation_writes_a_matching_pair(self):
        from dashboard import server as dashboard_server

        try:
            import cryptography  # noqa: F401
        except ImportError:
            self.skipTest("cryptography is not installed")

        with tempfile.TemporaryDirectory() as temp:
            with (
                mock.patch.object(dashboard_server, "BASE_DIR", Path(temp)),
                mock.patch.object(dashboard_server, "_local_ip", return_value="127.0.0.1"),
            ):
                self.assertTrue(dashboard_server._ensure_certs())
            certs = Path(temp) / "config" / "certs"
            self.assertTrue(dashboard_server._certificate_pair_valid(
                certs / "jarvis.key", certs / "jarvis.crt"
            ))


if __name__ == "__main__":
    unittest.main()
