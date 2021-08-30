import React from "react";
import {
    Button,
    Panel,
    FormGroup,
    FormControl,
    Form
} from "react-bootstrap";
import {connect} from "react-redux";
import {bindActionCreators} from "redux";
import * as ClassifyTokensInSentenceActions from "./ClassifyTokensInSentenceActions";
import "../Common/style.css"
import * as utils from "../Common/Utils"

class ClassifyTokensInSentenceForm extends React.Component {

    state = {
        sentence: null,
        processingStatus: "idle",
        classifiedTokens: []
    };

    getClassifiedTokens = () => {
        let items = []
        if (this.state.classifiedTokens.length !== 0)
            this.state.classifiedTokens.forEach((element) => {
                if (element[1] === "O") {
                    items.push(element[0])
                } else {
                    items.push(utils.getLabelForClass(element[0], element[1]))
                }
                items.push(" ")
            })
        return <div className="form-control" id="classified-tokens-container">
            {items}
        </div>
    }

    render() {
        return (
            <div>
                <Panel id="main-panel">
                    <Panel.Heading>
                        <Panel.Title>Token Classification</Panel.Title>
                    </Panel.Heading>
                    <Panel.Body>
                        <Form inline>
                            <FormGroup>
                                <FormControl
                                    type="text"
                                    id="sentence-input"
                                    value={this.state.sentence}
                                    placeholder="Enter sentence"
                                    onChange={(e) => {
                                        this.setState({sentence: e.target.value});
                                    }}
                                />
                                <Button
                                    bsStyle="primary"
                                    id="classification-btn"
                                    disabled={this.state.processingStatus !== "idle"
                                    || this.state.sentence == null
                                    || this.state.sentence.trim() === ""}
                                    onClick={() => {
                                        this.setState({processingStatus: "in_progress"});
                                        this.props.classifyTokensInSentence(this.state.sentence, (classifiedTokens) => {
                                            this.setState({
                                                processingStatus: "idle",
                                                classifiedTokens: JSON.parse(classifiedTokens)
                                            });
                                        })
                                    }}>
                                    Execute
                                </Button>
                            </FormGroup>
                        </Form>
                        <div>
                            {this.getClassifiedTokens()}
                        </div>
                    </Panel.Body>
                </Panel>
            </div>
        );
    }
}

export default connect(
    () => ({}),
    dispatch => ({
        classifyTokensInSentence: bindActionCreators(ClassifyTokensInSentenceActions.classifyTokensInSentence, dispatch)
    })
)(ClassifyTokensInSentenceForm)