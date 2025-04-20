import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Material

Page {
    id: home
    property StackView stack
    anchors.fill: parent
    background: Rectangle { color: "transparent" }

    ToolBar {
        height: 48
        width: parent.width
        contentItem: RowLayout {
            anchors.fill: parent
            spacing: 12

            ToolButton {
                text: "\u2630" // ícono hamburger
                Material.theme: Material.Dark
                background: Rectangle { color: "transparent" }
            }
            Label {
                text: "ChessPlayerAnalyzer"
                color: "white"
                font.pixelSize: 20
                Layout.alignment: Qt.AlignLeft | Qt.AlignVCenter
            }
        }
    }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 24

        Rectangle {
            id: card
            color: "#1e1e1e"
            radius: 12
            width: 420
            height: 180

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 24
                spacing: 16

                TextField {
                    id: userField
                    placeholderText: qsTr("Nombre de usuario")
                    color: "white"
                    placeholderTextColor: "#8c8c8c"
                    height: 40
                    font.pixelSize: 16
                    background: Rectangle {
                        color: "#191919"
                        radius: 6
                        border.color: root.accent
                        border.width: 1
                    }
                }

                Button {
                    text: qsTr("Investigar")
                    height: 40
                    font.pixelSize: 16
                    background: Rectangle { color: root.accent; radius: 6 }
                    onClicked: if (userField.text.trim().length > 0)
                                   stack.push(Qt.resolvedUrl("ResultsPage.qml"),
                                              { username: userField.text.trim() })
                }
            }
        }

        ListView {
            id: historyList
            width: 420
            height: 200
            model: historyModel
            clip: true
            delegate: ItemDelegate {
                width: ListView.view.width
                text: modelData
                onClicked: stack.push(Qt.resolvedUrl("ResultsPage.qml"), { username: modelData })
            }
        }

        ListModel { id: historyModel }
    }
}
