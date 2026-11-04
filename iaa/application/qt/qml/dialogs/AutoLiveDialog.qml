import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ".." as App
import "../controls"
import "../components/form"

// 自动演出对话框。
// 版面：左边固定宽度的标签栏、右边控件填满；分组顺序为
//   预设 → 演出方式（怎么打）→ 打什么 → 跑多久／休息 → 其他。
// 当前演出方式用不到的栏位一律隐藏；被模式固定的值写在模式下方的说明里。
Dialog {
    id: root
    modal: true
    title: App.Globals.t("task.auto_live")
    width: 560
    // 内容超出视窗时捲动，底部按钮固定
    height: Math.min(implicitHeight, Overlay.overlay ? Overlay.overlay.height - 48 : implicitHeight)
    anchors.centerIn: Overlay.overlay
    standardButtons: Dialog.NoButton

    property var runCtrl: null
    property var presets: []
    // 按过「开始」后，必填但留空的栏位才显示成红字
    property bool attempted: false

    readonly property string apKeepValue: root.runCtrl ? root.runCtrl.apKeepValue : ""
    readonly property string songKeepValue: root.runCtrl ? root.runCtrl.songKeepValue : ""
    readonly property bool lightScheme: Application.styleHints.colorScheme === Qt.Light
    readonly property color errorColor: lightScheme ? "#C42B1C" : "#FF99A4"
    readonly property color hintColor: palette.placeholderText
    readonly property int labelWidth: 72
    readonly property int controlHeight: 32

    function defaultPayload() {
        return {
            playMode: "game_auto",
            apMultiplier: root.apKeepValue,
            latencyCompensationMs: "0",
            loopMode: "list",
            songName: root.songKeepValue,
            countMode: "specify",
            count: "10",
            durationText: "",
            untilTime: "",
            restMode: "none",
            restEveryRounds: "30",
            restEveryText: "",
            restForText: "",
            debugEnabled: false,
            autoSetUnit: false
        }
    }

    property var formData: defaultPayload()
    // 切到脚本自动／长按中央会把 AP 固定成 0；切回游戏自动时还原
    property string _gameAp: root.apKeepValue

    function updateField(key, value) {
        formData = Object.assign({}, formData, { [key]: value })
    }

    function applyPreset(preset) {
        var base = defaultPayload()
        for (var k in base) {
            if (preset[k] !== undefined && preset[k] !== null && preset[k] !== "")
                base[k] = preset[k]
        }
        if (!preset.songName) base.songName = root.songKeepValue
        attempted = false
        formData = base
    }

    // ── 演出方式 ──────────────────────────────────────────
    readonly property var playModeOptions: [
        { text: App.Globals.t("auto_live.play.game_auto"), value: "game_auto" },
        { text: App.Globals.t("auto_live.play.script_auto"), value: "script_auto" },
        { text: App.Globals.t("auto_live.play.hold_center"), value: "hold_center" }
    ]
    readonly property bool isGame: formData.playMode === "game_auto"
    readonly property bool isScript: formData.playMode === "script_auto"
    readonly property bool isHold: formData.playMode === "hold_center"

    function setPlayMode(mode) {
        if (mode === formData.playMode) return
        var patch = { playMode: mode }
        if (formData.playMode === "game_auto") _gameAp = formData.apMultiplier
        if (mode === "game_auto") {
            patch.apMultiplier = _gameAp
        } else {
            patch.apMultiplier = "0"
        }
        if (mode === "hold_center") {
            patch.loopMode = "single"
            patch.countMode = "all"     // 长按中央预设「不限」
        } else if (runOptionsFor(mode).map(function(o) { return o.value }).indexOf(formData.countMode) < 0) {
            patch.countMode = "specify"
        }
        attempted = false
        formData = Object.assign({}, formData, patch)
    }

    function modeNotice() {
        if (isHold) return App.Globals.t("auto_live.notice.hold_center")
        if (isScript) return App.Globals.t("auto_live.notice.ap_locked_zero")
        return ""
    }

    // ── 下拉选项 ──────────────────────────────────────────
    function runOptionsFor(mode) {
        if (mode === "game_auto") return [
            { text: App.Globals.t("auto_live.count.specify"), value: "specify" },
            { text: App.Globals.t("auto_live.count.all"), value: "all" }
        ]
        if (mode === "script_auto") return [
            { text: App.Globals.t("auto_live.count.unlimited"), value: "all" },
            { text: App.Globals.t("auto_live.count.specify"), value: "specify" }
        ]
        return [
            { text: App.Globals.t("auto_live.count.unlimited"), value: "all" },
            { text: App.Globals.t("auto_live.count.specify"), value: "specify" },
            { text: App.Globals.t("auto_live.count.duration"), value: "duration" },
            { text: App.Globals.t("auto_live.count.until"), value: "until" }
        ]
    }
    readonly property var runOptions: runOptionsFor(formData.playMode)

    readonly property var loopOptions: [
        { text: App.Globals.t("auto_live.loop.single"), value: "single" },
        { text: App.Globals.t("auto_live.loop.list"), value: "list" },
        { text: App.Globals.t("auto_live.loop.random"), value: "random" }
    ]

    readonly property var restOptions: [
        { text: App.Globals.t("auto_live.rest.none"), value: "none" },
        { text: App.Globals.t("auto_live.rest.rounds"), value: "rounds" },
        { text: App.Globals.t("auto_live.rest.time"), value: "time" }
    ]

    function indexIn(options, value) {
        for (var i = 0; i < options.length; i++)
            if (options[i].value === value) return i
        return 0
    }

    function apMultiplierLabel(value) {
        if (value === root.apKeepValue) return App.Globals.t("auto_live.ap.keep")
        if (value === "maximum") return App.Globals.t("auto_live.ap.maximum")
        return value
    }

    function apMultiplierValue(label) {
        if (label === App.Globals.t("auto_live.ap.keep")) return root.apKeepValue
        if (label === App.Globals.t("auto_live.ap.maximum")) return "maximum"
        return label
    }

    function apMultiplierOptions() {
        return [
            App.Globals.t("auto_live.ap.keep"),
            App.Globals.t("auto_live.ap.maximum"),
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
        ]
    }

    function songNameLabel(value) {
        return value === root.songKeepValue ? App.Globals.t("auto_live.song.keep") : value
    }

    function songNameValue(label) {
        return label === App.Globals.t("auto_live.song.keep") ? root.songKeepValue : label
    }

    function songNameOptions() {
        return [App.Globals.t("auto_live.song.keep"), "メルト", "独りんぼエンヴィー"]
    }

    function presetNameLabel(name) {
        return root.runCtrl.autoLivePresetLabel(name)
    }

    // ── 格式检查（与后端 parse_duration / parse_until 规则一致）──────────
    // 每个检查返回 { ok, text }：text 是栏位下方的提示；ok=false 时显示成红字。

    // 「2h」「90m」「1h30m」「45」（纯数字＝分钟）
    function parseDuration(text) {
        var s = String(text || "").trim().toLowerCase().replace(/ /g, "")
        if (!s) return null
        var h = 0, m = 0
        if (/^\d+$/.test(s)) {
            m = parseInt(s, 10)
        } else {
            var mm = s.match(/^(?:(\d+)h)?(?:(\d+)m)?$/)
            if (!mm || !(mm[1] || mm[2])) return -1
            h = parseInt(mm[1] || "0", 10)
            m = parseInt(mm[2] || "0", 10)
        }
        var total = h * 60 + m
        return total > 0 ? total : -1
    }

    function durationLabel(minutes) {
        var h = Math.floor(minutes / 60), m = minutes % 60
        var parts = []
        if (h) parts.push(h + " " + App.Globals.t("auto_live.unit.hours"))
        if (m) parts.push(m + " " + App.Globals.t("auto_live.unit.minutes"))
        return "= " + parts.join(" ")
    }

    function checkDuration(text) {
        var minutes = parseDuration(text)
        if (minutes === null)
            return { ok: !attempted, text: App.Globals.t("auto_live.hint.duration_example") }
        if (minutes < 0) return { ok: false, text: App.Globals.t("auto_live.hint.time_format") }
        return { ok: true, text: durationLabel(minutes) }
    }

    function checkRestFor(text) {
        var s = String(text || "").trim()
        if (!s) return { ok: !attempted, text: App.Globals.t("auto_live.hint.rest_for_empty") }
        if (s === "0") return { ok: true, text: App.Globals.t("auto_live.hint.rest_for_zero") }
        return checkDuration(s)
    }

    function checkUntil(text) {
        var s = String(text || "").trim()
        if (!s) return { ok: !attempted, text: App.Globals.t("auto_live.hint.until") }
        var mm = s.match(/^(\d{1,2}):(\d{2})$/)
        if (!mm || parseInt(mm[1], 10) > 23 || parseInt(mm[2], 10) > 59)
            return { ok: false, text: App.Globals.t("auto_live.hint.until_format") }
        return { ok: true, text: App.Globals.t("auto_live.hint.until") }
    }

    function checkPositiveInt(text) {
        var s = String(text || "").trim()
        if (/^\d+$/.test(s) && parseInt(s, 10) > 0) return { ok: true, text: "" }
        if (!s && !attempted) return { ok: true, text: "" }
        return { ok: false, text: App.Globals.t("auto_live.hint.positive_int") }
    }

    function checkLatency(text) {
        var s = String(text || "").trim()
        if (/^\d+$/.test(s) && parseInt(s, 10) <= 2000)
            return { ok: true, text: App.Globals.t("auto_live.hint.latency") }
        return { ok: false, text: App.Globals.t("auto_live.hint.latency") }
    }

    readonly property var countCheck: checkPositiveInt(formData.count)
    readonly property var durationCheck: checkDuration(formData.durationText)
    readonly property var untilCheck: checkUntil(formData.untilTime)
    readonly property var restRoundsCheck: checkPositiveInt(formData.restEveryRounds)
    readonly property var restTimeCheck: checkDuration(formData.restEveryText)
    readonly property var restForCheck: checkRestFor(formData.restForText)
    readonly property var latencyCheck: checkLatency(formData.latencyCompensationMs)

    // 只检查当前看得到的栏位；必填留空在这里也算错
    function hasErrors() {
        var f = formData
        if (isScript && !latencyCheck.ok) return true
        if (f.countMode === "specify" && !(/^\d+$/.test(String(f.count).trim()) && parseInt(f.count, 10) > 0)) return true
        if (isHold && f.countMode === "duration" && !(parseDuration(f.durationText) > 0)) return true
        if (isHold && f.countMode === "until" && !checkUntil(f.untilTime).ok) return true
        if (isHold && f.restMode !== "none") {
            if (f.restMode === "rounds" && !(/^\d+$/.test(String(f.restEveryRounds).trim()) && parseInt(f.restEveryRounds, 10) > 0)) return true
            if (f.restMode === "time" && !(parseDuration(f.restEveryText) > 0)) return true
            var rf = String(f.restForText || "").trim()
            if (rf !== "0" && !(parseDuration(rf) > 0)) return true
        }
        return false
    }

    onOpened: {
        presets = JSON.parse(root.runCtrl.builtinAutoPresetsJson())
        attempted = false
        formData = defaultPayload()
    }

    // ── 可重复使用的小元件 ────────────────────────────────────
    component FieldLabel: Label {
        Layout.preferredWidth: root.labelWidth
        Layout.preferredHeight: root.controlHeight
        Layout.alignment: Qt.AlignTop
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
        opacity: 0.8
    }

    component Hint: Label {
        property var check: null
        Layout.fillWidth: true
        // 固定一行高度：提示文字变长变短时下方栏位不会跳动
        Layout.preferredHeight: 16
        font.pixelSize: 12
        elide: Text.ElideRight
        text: check ? check.text : ""
        color: check && !check.ok ? root.errorColor : root.hintColor
    }

    component Divider: Rectangle {
        Layout.columnSpan: 2
        Layout.fillWidth: true
        Layout.topMargin: 2
        Layout.bottomMargin: 2
        implicitHeight: 1
        color: root.palette.text
        opacity: 0.1
    }

    component OptionSelect: Select {
        property var options: []
        property string current: ""
        signal picked(string value)
        Layout.fillWidth: true
        model: options
        textRole: "text"
        valueRole: "value"
        currentIndex: root.indexIn(options, current)
        onActivated: picked(options[currentIndex].value)
    }

    contentItem: ScrollView {
        id: scroll
        clip: true
        contentWidth: availableWidth
        implicitHeight: grid.implicitHeight

        GridLayout {
            id: grid
            width: scroll.availableWidth
            columns: 2
            columnSpacing: 14
            rowSpacing: 12

            // ── 预设 ──
            FieldLabel { text: App.Globals.t("auto_live.preset") }
            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: root.presets
                    delegate: Button {
                        required property var modelData
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        text: root.presetNameLabel(modelData.name)
                        onClicked: root.applyPreset(modelData)
                    }
                }
                Button {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    text: App.Globals.t("auto_live.preset.last")
                    onClicked: {
                        var raw = root.runCtrl.lastAutoPresetJson()
                        if (!raw) {
                            App.Notice.show("error", App.Globals.t("auto_live.notice.no_last_preset"))
                            return
                        }
                        root.applyPreset(JSON.parse(raw))
                    }
                }
            }

            Divider {}

            // ── 怎么打 ──
            FieldLabel { text: App.Globals.t("auto_live.play_mode") }
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 8
                SegmentedButton {
                    Layout.fillWidth: true
                    equalWidth: true
                    model: root.playModeOptions
                    textRole: "text"
                    valueRole: "value"
                    value: root.formData.playMode
                    onActivated: function(index, value) { root.setPlayMode(value) }
                }
                FormNotice {
                    visible: !root.isGame
                    style: "info"
                    content: root.modeNotice()
                }
            }

            FieldLabel { visible: root.isGame; text: App.Globals.t("auto_live.ap_multiplier") }
            Select {
                visible: root.isGame
                Layout.fillWidth: true
                model: root.apMultiplierOptions()
                currentIndex: Math.max(0, model.indexOf(root.apMultiplierLabel(root.formData.apMultiplier)))
                onActivated: root.updateField("apMultiplier", root.apMultiplierValue(model[currentIndex]))
            }

            FieldLabel { visible: root.isScript; text: App.Globals.t("auto_live.latency") }
            ColumnLayout {
                visible: root.isScript
                Layout.fillWidth: true
                spacing: 4
                RowLayout {
                    Layout.fillWidth: true
                    TextField {
                        Layout.fillWidth: true
                        text: root.formData.latencyCompensationMs
                        onTextEdited: root.updateField("latencyCompensationMs", text)
                    }
                    Label { text: App.Globals.t("auto_live.unit.ms"); opacity: 0.8 }
                }
                Hint { check: root.latencyCheck }
            }

            Divider {}

            // ── 打什么 ──
            FieldLabel { visible: !root.isHold; text: App.Globals.t("auto_live.loop_mode") }
            OptionSelect {
                visible: !root.isHold
                options: root.loopOptions
                current: root.formData.loopMode
                onPicked: function(value) { root.updateField("loopMode", value) }
            }

            FieldLabel {
                visible: root.isHold || root.formData.loopMode === "single"
                text: App.Globals.t("auto_live.song_name")
            }
            ComboBox {
                visible: root.isHold || root.formData.loopMode === "single"
                Layout.fillWidth: true
                model: root.songNameOptions()
                editable: true
                currentIndex: Math.max(0, model.indexOf(root.songNameLabel(root.formData.songName)))
                onActivated: root.updateField("songName", root.songNameValue(model[currentIndex]))
                onEditTextChanged: {
                    var v = root.songNameValue(editText)
                    if (v !== root.formData.songName) root.updateField("songName", v)
                }
            }

            Divider {}

            // ── 跑多久 ──
            FieldLabel { text: App.Globals.t("auto_live.count") }
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 8
                OptionSelect {
                    options: root.runOptions
                    current: root.formData.countMode
                    onPicked: function(value) { root.updateField("countMode", value) }
                }
                ColumnLayout {
                    visible: root.formData.countMode === "specify"
                    Layout.fillWidth: true
                    spacing: 4
                    RowLayout {
                        Layout.fillWidth: true
                        TextField {
                            Layout.fillWidth: true
                            text: root.formData.count
                            onTextEdited: root.updateField("count", text)
                        }
                        Label { text: App.Globals.t("auto_live.unit.rounds"); opacity: 0.8 }
                    }
                    Hint { visible: !root.countCheck.ok; check: root.countCheck }
                }
                ColumnLayout {
                    visible: root.isHold && root.formData.countMode === "duration"
                    Layout.fillWidth: true
                    spacing: 4
                    TextField {
                        Layout.fillWidth: true
                        text: root.formData.durationText
                        onTextEdited: root.updateField("durationText", text)
                    }
                    Hint { check: root.durationCheck }
                }
                ColumnLayout {
                    visible: root.isHold && root.formData.countMode === "until"
                    Layout.fillWidth: true
                    spacing: 4
                    TextField {
                        Layout.fillWidth: true
                        text: root.formData.untilTime
                        onTextEdited: root.updateField("untilTime", text)
                    }
                    Hint { check: root.untilCheck }
                }
            }

            // ── 休息（目前只有长按中央）──
            FieldLabel { visible: root.isHold; text: App.Globals.t("auto_live.rest") }
            ColumnLayout {
                visible: root.isHold
                Layout.fillWidth: true
                spacing: 8
                OptionSelect {
                    options: root.restOptions
                    current: root.formData.restMode
                    onPicked: function(value) { root.updateField("restMode", value) }
                }
                GridLayout {
                    visible: root.formData.restMode !== "none"
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 16
                    rowSpacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        Label { text: App.Globals.t("auto_live.rest.every"); opacity: 0.8 }
                        TextField {
                            visible: root.formData.restMode === "rounds"
                            Layout.fillWidth: true
                            text: root.formData.restEveryRounds
                            onTextEdited: root.updateField("restEveryRounds", text)
                        }
                        Label {
                            visible: root.formData.restMode === "rounds"
                            text: App.Globals.t("auto_live.unit.rounds")
                            opacity: 0.8
                        }
                        TextField {
                            visible: root.formData.restMode === "time"
                            Layout.fillWidth: true
                            text: root.formData.restEveryText
                            onTextEdited: root.updateField("restEveryText", text)
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredWidth: 1
                        Label { text: App.Globals.t("auto_live.rest.for"); opacity: 0.8 }
                        TextField {
                            Layout.fillWidth: true
                            text: root.formData.restForText
                            onTextEdited: root.updateField("restForText", text)
                        }
                    }
                    Hint {
                        Layout.preferredWidth: 1
                        check: root.formData.restMode === "time" ? root.restTimeCheck
                             : (root.restRoundsCheck.ok ? null : root.restRoundsCheck)
                    }
                    Hint {
                        Layout.preferredWidth: 1
                        check: root.restForCheck
                    }
                }
            }

            Divider {}

            // ── 其他 ──
            Item { Layout.preferredWidth: root.labelWidth; implicitHeight: 1 }
            RowLayout {
                Layout.fillWidth: true
                spacing: 20
                CheckBox {
                    text: App.Globals.t("auto_live.auto_set_unit")
                    checked: root.formData.autoSetUnit
                    onToggled: root.updateField("autoSetUnit", checked)
                }
                CheckBox {
                    visible: root.isScript
                    text: App.Globals.t("auto_live.debug_display")
                    checked: root.formData.debugEnabled
                    onToggled: root.updateField("debugEnabled", checked)
                }
                Item { Layout.fillWidth: true }
            }
        }
    }

    footer: Item {
        implicitHeight: buttons.implicitHeight + 24
        RowLayout {
            id: buttons
            anchors.right: parent.right
            anchors.rightMargin: 24
            anchors.verticalCenter: parent.verticalCenter
            spacing: 8
            Button {
                Layout.preferredWidth: 96
                text: App.Globals.t("common.cancel")
                onClicked: root.close()
            }
            Button {
                Layout.preferredWidth: 96
                text: App.Globals.t("common.start")
                highlighted: true
                onClicked: {
                    root.attempted = true
                    if (root.hasErrors()) {
                        App.Notice.show("error", App.Globals.t("auto_live.hint.fix_errors"))
                        return
                    }
                    try {
                        root.runCtrl.runAutoLive(JSON.stringify(root.formData))
                        root.close()
                    } catch (error) {
                        App.Notice.show("error", String(error))
                    }
                }
            }
        }
    }
}
