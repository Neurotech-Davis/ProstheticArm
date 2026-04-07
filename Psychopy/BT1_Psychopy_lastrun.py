#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on Thu Feb  5 16:50:10 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'BT1_Psychopy'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 900]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/tonysaldana/Downloads/BT1_Psychopy_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('Intro_Key_Resp') is None:
        # initialise Intro_Key_Resp
        Intro_Key_Resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Intro_Key_Resp',
        )
    # create speaker 'PT_EC_S1'
    deviceManager.addDevice(
        deviceName='PT_EC_S1',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    # create speaker 'PT_EO_S2'
    deviceManager.addDevice(
        deviceName='PT_EO_S2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('Transition_Key_Resp') is None:
        # initialise Transition_Key_Resp
        Transition_Key_Resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Transition_Key_Resp',
        )
    # create speaker 'RT_EC'
    deviceManager.addDevice(
        deviceName='RT_EC',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    # create speaker 'RT_EO'
    deviceManager.addDevice(
        deviceName='RT_EO',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('Rest_Key_Resp') is None:
        # initialise Rest_Key_Resp
        Rest_Key_Resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Rest_Key_Resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Introduction" ---
    Introduction_text = visual.TextStim(win=win, name='Introduction_text',
        text='In the following Experiement you will be opening and closing your eyes. When you hear the noise you will close your eyes and when you hear it again you will open them.\nTo begin, put yourself in a relaxed state.\nTry to remain still and not clench your jaw.\nTry not to blink during the experiment ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Practice_Trial_Text = visual.TextStim(win=win, name='Practice_Trial_Text',
        text='We will now begin the practice session.\nYou will focus on the dot in the middle.\nTo begin the practice session press the space bar.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Intro_Key_Resp = keyboard.Keyboard(deviceName='Intro_Key_Resp')
    
    # --- Initialize components for Routine "Practice_Session" ---
    PT_Dot_EC = visual.ImageStim(
        win=win,
        name='PT_Dot_EC', 
        image='/Users/tonysaldana/Downloads/Neurotech 25-26/Beginner Project/Black_Dot.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    PT_EC_S1 = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='PT_EC_S1',    name='PT_EC_S1'
    )
    PT_EC_S1.setVolume(1.0)
    PT_Dot_EO = visual.ImageStim(
        win=win,
        name='PT_Dot_EO', 
        image='/Users/tonysaldana/Downloads/Neurotech 25-26/Beginner Project/Black_Dot.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    PT_EO_S2 = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='PT_EO_S2',    name='PT_EO_S2'
    )
    PT_EO_S2.setVolume(1.0)
    
    # --- Initialize components for Routine "Intermission" ---
    Transition_Text = visual.TextStim(win=win, name='Transition_Text',
        text='We will know begin the actual trial sessions for the experiments. Press space when you are ready to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Transition_Key_Resp = keyboard.Keyboard(deviceName='Transition_Key_Resp')
    
    # --- Initialize components for Routine "Trial_Session" ---
    RT_EC = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='RT_EC',    name='RT_EC'
    )
    RT_EC.setVolume(1.0)
    RT_EO = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='RT_EO',    name='RT_EO'
    )
    RT_EO.setVolume(1.0)
    RT_Dot_EC = visual.ImageStim(
        win=win,
        name='RT_Dot_EC', 
        image='/Users/tonysaldana/Downloads/Neurotech 25-26/Beginner Project/Black_Dot.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    RT_Dot_EO = visual.ImageStim(
        win=win,
        name='RT_Dot_EO', 
        image='/Users/tonysaldana/Downloads/Neurotech 25-26/Beginner Project/Black_Dot.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "rest" ---
    Rest_text = visual.TextStim(win=win, name='Rest_text',
        text='Rest your eyes as long as you need. \nPress space when you are ready to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    Rest_Key_Resp = keyboard.Keyboard(deviceName='Rest_Key_Resp')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Introduction" ---
    # create an object to store info about Routine Introduction
    Introduction = data.Routine(
        name='Introduction',
        components=[Introduction_text, Practice_Trial_Text, Intro_Key_Resp],
    )
    Introduction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Intro_Key_Resp
    Intro_Key_Resp.keys = []
    Intro_Key_Resp.rt = []
    _Intro_Key_Resp_allKeys = []
    # store start times for Introduction
    Introduction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Introduction.tStart = globalClock.getTime(format='float')
    Introduction.status = STARTED
    thisExp.addData('Introduction.started', Introduction.tStart)
    Introduction.maxDuration = None
    # keep track of which components have finished
    IntroductionComponents = Introduction.components
    for thisComponent in Introduction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Introduction" ---
    Introduction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Introduction_text* updates
        
        # if Introduction_text is starting this frame...
        if Introduction_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Introduction_text.frameNStart = frameN  # exact frame index
            Introduction_text.tStart = t  # local t and not account for scr refresh
            Introduction_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Introduction_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Introduction_text.started')
            # update status
            Introduction_text.status = STARTED
            Introduction_text.setAutoDraw(True)
        
        # if Introduction_text is active this frame...
        if Introduction_text.status == STARTED:
            # update params
            pass
        
        # if Introduction_text is stopping this frame...
        if Introduction_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Introduction_text.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                Introduction_text.tStop = t  # not accounting for scr refresh
                Introduction_text.tStopRefresh = tThisFlipGlobal  # on global time
                Introduction_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Introduction_text.stopped')
                # update status
                Introduction_text.status = FINISHED
                Introduction_text.setAutoDraw(False)
        
        # *Practice_Trial_Text* updates
        
        # if Practice_Trial_Text is starting this frame...
        if Practice_Trial_Text.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            Practice_Trial_Text.frameNStart = frameN  # exact frame index
            Practice_Trial_Text.tStart = t  # local t and not account for scr refresh
            Practice_Trial_Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Practice_Trial_Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Practice_Trial_Text.started')
            # update status
            Practice_Trial_Text.status = STARTED
            Practice_Trial_Text.setAutoDraw(True)
        
        # if Practice_Trial_Text is active this frame...
        if Practice_Trial_Text.status == STARTED:
            # update params
            pass
        
        # *Intro_Key_Resp* updates
        waitOnFlip = False
        
        # if Intro_Key_Resp is starting this frame...
        if Intro_Key_Resp.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            Intro_Key_Resp.frameNStart = frameN  # exact frame index
            Intro_Key_Resp.tStart = t  # local t and not account for scr refresh
            Intro_Key_Resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Intro_Key_Resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Intro_Key_Resp.started')
            # update status
            Intro_Key_Resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Intro_Key_Resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Intro_Key_Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Intro_Key_Resp.status == STARTED and not waitOnFlip:
            theseKeys = Intro_Key_Resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Intro_Key_Resp_allKeys.extend(theseKeys)
            if len(_Intro_Key_Resp_allKeys):
                Intro_Key_Resp.keys = _Intro_Key_Resp_allKeys[-1].name  # just the last key pressed
                Intro_Key_Resp.rt = _Intro_Key_Resp_allKeys[-1].rt
                Intro_Key_Resp.duration = _Intro_Key_Resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Introduction,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Introduction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Introduction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Introduction" ---
    for thisComponent in Introduction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Introduction
    Introduction.tStop = globalClock.getTime(format='float')
    Introduction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Introduction.stopped', Introduction.tStop)
    # check responses
    if Intro_Key_Resp.keys in ['', [], None]:  # No response was made
        Intro_Key_Resp.keys = None
    thisExp.addData('Intro_Key_Resp.keys',Intro_Key_Resp.keys)
    if Intro_Key_Resp.keys != None:  # we had a response
        thisExp.addData('Intro_Key_Resp.rt', Intro_Key_Resp.rt)
        thisExp.addData('Intro_Key_Resp.duration', Intro_Key_Resp.duration)
    thisExp.nextEntry()
    # the Routine "Introduction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Practice_Loop = data.TrialHandler2(
        name='Practice_Loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(Practice_Loop)  # add the loop to the experiment
    thisPractice_Loop = Practice_Loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_Loop.rgb)
    if thisPractice_Loop != None:
        for paramName in thisPractice_Loop:
            globals()[paramName] = thisPractice_Loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_Loop in Practice_Loop:
        Practice_Loop.status = STARTED
        if hasattr(thisPractice_Loop, 'status'):
            thisPractice_Loop.status = STARTED
        currentLoop = Practice_Loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_Loop.rgb)
        if thisPractice_Loop != None:
            for paramName in thisPractice_Loop:
                globals()[paramName] = thisPractice_Loop[paramName]
        
        # --- Prepare to start Routine "Practice_Session" ---
        # create an object to store info about Routine Practice_Session
        Practice_Session = data.Routine(
            name='Practice_Session',
            components=[PT_Dot_EC, PT_EC_S1, PT_Dot_EO, PT_EO_S2],
        )
        Practice_Session.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        PT_EC_S1.setSound('/Users/tonysaldana/Downloads/Neurotech 25-26/Beginner Project/beep-07a.mp3', secs=1.0, hamming=True)
        PT_EC_S1.setVolume(1.0, log=False)
        PT_EC_S1.seek(0)
        PT_EO_S2.setSound('/Users/tonysaldana/Downloads/Neurotech 25-26/Beginner Project/beep-07a.mp3', secs=1.0, hamming=True)
        PT_EO_S2.setVolume(1.0, log=False)
        PT_EO_S2.seek(0)
        # store start times for Practice_Session
        Practice_Session.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Practice_Session.tStart = globalClock.getTime(format='float')
        Practice_Session.status = STARTED
        thisExp.addData('Practice_Session.started', Practice_Session.tStart)
        Practice_Session.maxDuration = None
        # keep track of which components have finished
        Practice_SessionComponents = Practice_Session.components
        for thisComponent in Practice_Session.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Practice_Session" ---
        Practice_Session.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 34.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_Loop, 'status') and thisPractice_Loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *PT_Dot_EC* updates
            
            # if PT_Dot_EC is starting this frame...
            if PT_Dot_EC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                PT_Dot_EC.frameNStart = frameN  # exact frame index
                PT_Dot_EC.tStart = t  # local t and not account for scr refresh
                PT_Dot_EC.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(PT_Dot_EC, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'PT_Dot_EC.started')
                # update status
                PT_Dot_EC.status = STARTED
                PT_Dot_EC.setAutoDraw(True)
            
            # if PT_Dot_EC is active this frame...
            if PT_Dot_EC.status == STARTED:
                # update params
                pass
            
            # if PT_Dot_EC is stopping this frame...
            if PT_Dot_EC.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > PT_Dot_EC.tStartRefresh + 17.0-frameTolerance:
                    # keep track of stop time/frame for later
                    PT_Dot_EC.tStop = t  # not accounting for scr refresh
                    PT_Dot_EC.tStopRefresh = tThisFlipGlobal  # on global time
                    PT_Dot_EC.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'PT_Dot_EC.stopped')
                    # update status
                    PT_Dot_EC.status = FINISHED
                    PT_Dot_EC.setAutoDraw(False)
            
            # *PT_EC_S1* updates
            
            # if PT_EC_S1 is starting this frame...
            if PT_EC_S1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                PT_EC_S1.frameNStart = frameN  # exact frame index
                PT_EC_S1.tStart = t  # local t and not account for scr refresh
                PT_EC_S1.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('PT_EC_S1.started', tThisFlipGlobal)
                # update status
                PT_EC_S1.status = STARTED
                PT_EC_S1.play(when=win)  # sync with win flip
            
            # if PT_EC_S1 is stopping this frame...
            if PT_EC_S1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > PT_EC_S1.tStartRefresh + 1.0-frameTolerance or PT_EC_S1.isFinished:
                    # keep track of stop time/frame for later
                    PT_EC_S1.tStop = t  # not accounting for scr refresh
                    PT_EC_S1.tStopRefresh = tThisFlipGlobal  # on global time
                    PT_EC_S1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'PT_EC_S1.stopped')
                    # update status
                    PT_EC_S1.status = FINISHED
                    PT_EC_S1.stop()
            
            # *PT_Dot_EO* updates
            
            # if PT_Dot_EO is starting this frame...
            if PT_Dot_EO.status == NOT_STARTED and tThisFlip >= 17.0-frameTolerance:
                # keep track of start time/frame for later
                PT_Dot_EO.frameNStart = frameN  # exact frame index
                PT_Dot_EO.tStart = t  # local t and not account for scr refresh
                PT_Dot_EO.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(PT_Dot_EO, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'PT_Dot_EO.started')
                # update status
                PT_Dot_EO.status = STARTED
                PT_Dot_EO.setAutoDraw(True)
            
            # if PT_Dot_EO is active this frame...
            if PT_Dot_EO.status == STARTED:
                # update params
                pass
            
            # if PT_Dot_EO is stopping this frame...
            if PT_Dot_EO.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > PT_Dot_EO.tStartRefresh + 17.0-frameTolerance:
                    # keep track of stop time/frame for later
                    PT_Dot_EO.tStop = t  # not accounting for scr refresh
                    PT_Dot_EO.tStopRefresh = tThisFlipGlobal  # on global time
                    PT_Dot_EO.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'PT_Dot_EO.stopped')
                    # update status
                    PT_Dot_EO.status = FINISHED
                    PT_Dot_EO.setAutoDraw(False)
            
            # *PT_EO_S2* updates
            
            # if PT_EO_S2 is starting this frame...
            if PT_EO_S2.status == NOT_STARTED and tThisFlip >= 17.0-frameTolerance:
                # keep track of start time/frame for later
                PT_EO_S2.frameNStart = frameN  # exact frame index
                PT_EO_S2.tStart = t  # local t and not account for scr refresh
                PT_EO_S2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('PT_EO_S2.started', tThisFlipGlobal)
                # update status
                PT_EO_S2.status = STARTED
                PT_EO_S2.play(when=win)  # sync with win flip
            
            # if PT_EO_S2 is stopping this frame...
            if PT_EO_S2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > PT_EO_S2.tStartRefresh + 1.0-frameTolerance or PT_EO_S2.isFinished:
                    # keep track of stop time/frame for later
                    PT_EO_S2.tStop = t  # not accounting for scr refresh
                    PT_EO_S2.tStopRefresh = tThisFlipGlobal  # on global time
                    PT_EO_S2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'PT_EO_S2.stopped')
                    # update status
                    PT_EO_S2.status = FINISHED
                    PT_EO_S2.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Practice_Session,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Practice_Session.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Practice_Session.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Practice_Session" ---
        for thisComponent in Practice_Session.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Practice_Session
        Practice_Session.tStop = globalClock.getTime(format='float')
        Practice_Session.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Practice_Session.stopped', Practice_Session.tStop)
        PT_EC_S1.pause()  # ensure sound has stopped at end of Routine
        PT_EO_S2.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Practice_Session.maxDurationReached:
            routineTimer.addTime(-Practice_Session.maxDuration)
        elif Practice_Session.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-34.000000)
        # mark thisPractice_Loop as finished
        if hasattr(thisPractice_Loop, 'status'):
            thisPractice_Loop.status = FINISHED
        # if awaiting a pause, pause now
        if Practice_Loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            Practice_Loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'Practice_Loop'
    Practice_Loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if Practice_Loop.trialList in ([], [None], None):
        params = []
    else:
        params = Practice_Loop.trialList[0].keys()
    # save data for this loop
    Practice_Loop.saveAsText(filename + '_Practice_Loop.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "Intermission" ---
    # create an object to store info about Routine Intermission
    Intermission = data.Routine(
        name='Intermission',
        components=[Transition_Text, Transition_Key_Resp],
    )
    Intermission.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Transition_Key_Resp
    Transition_Key_Resp.keys = []
    Transition_Key_Resp.rt = []
    _Transition_Key_Resp_allKeys = []
    # store start times for Intermission
    Intermission.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intermission.tStart = globalClock.getTime(format='float')
    Intermission.status = STARTED
    thisExp.addData('Intermission.started', Intermission.tStart)
    Intermission.maxDuration = None
    # keep track of which components have finished
    IntermissionComponents = Intermission.components
    for thisComponent in Intermission.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Intermission" ---
    Intermission.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Transition_Text* updates
        
        # if Transition_Text is starting this frame...
        if Transition_Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Transition_Text.frameNStart = frameN  # exact frame index
            Transition_Text.tStart = t  # local t and not account for scr refresh
            Transition_Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Transition_Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Transition_Text.started')
            # update status
            Transition_Text.status = STARTED
            Transition_Text.setAutoDraw(True)
        
        # if Transition_Text is active this frame...
        if Transition_Text.status == STARTED:
            # update params
            pass
        
        # *Transition_Key_Resp* updates
        waitOnFlip = False
        
        # if Transition_Key_Resp is starting this frame...
        if Transition_Key_Resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Transition_Key_Resp.frameNStart = frameN  # exact frame index
            Transition_Key_Resp.tStart = t  # local t and not account for scr refresh
            Transition_Key_Resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Transition_Key_Resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Transition_Key_Resp.started')
            # update status
            Transition_Key_Resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Transition_Key_Resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Transition_Key_Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Transition_Key_Resp.status == STARTED and not waitOnFlip:
            theseKeys = Transition_Key_Resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Transition_Key_Resp_allKeys.extend(theseKeys)
            if len(_Transition_Key_Resp_allKeys):
                Transition_Key_Resp.keys = _Transition_Key_Resp_allKeys[-1].name  # just the last key pressed
                Transition_Key_Resp.rt = _Transition_Key_Resp_allKeys[-1].rt
                Transition_Key_Resp.duration = _Transition_Key_Resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Intermission,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Intermission.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Intermission.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intermission" ---
    for thisComponent in Intermission.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intermission
    Intermission.tStop = globalClock.getTime(format='float')
    Intermission.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intermission.stopped', Intermission.tStop)
    # check responses
    if Transition_Key_Resp.keys in ['', [], None]:  # No response was made
        Transition_Key_Resp.keys = None
    thisExp.addData('Transition_Key_Resp.keys',Transition_Key_Resp.keys)
    if Transition_Key_Resp.keys != None:  # we had a response
        thisExp.addData('Transition_Key_Resp.rt', Transition_Key_Resp.rt)
        thisExp.addData('Transition_Key_Resp.duration', Transition_Key_Resp.duration)
    thisExp.nextEntry()
    # the Routine "Intermission" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Block_Loop = data.TrialHandler2(
        name='Block_Loop',
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(Block_Loop)  # add the loop to the experiment
    thisBlock_Loop = Block_Loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_Loop.rgb)
    if thisBlock_Loop != None:
        for paramName in thisBlock_Loop:
            globals()[paramName] = thisBlock_Loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock_Loop in Block_Loop:
        Block_Loop.status = STARTED
        if hasattr(thisBlock_Loop, 'status'):
            thisBlock_Loop.status = STARTED
        currentLoop = Block_Loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock_Loop.rgb)
        if thisBlock_Loop != None:
            for paramName in thisBlock_Loop:
                globals()[paramName] = thisBlock_Loop[paramName]
        
        # set up handler to look after randomisation of conditions etc
        Trial_Loop = data.TrialHandler2(
            name='Trial_Loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(Trial_Loop)  # add the loop to the experiment
        thisTrial_Loop = Trial_Loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_Loop.rgb)
        if thisTrial_Loop != None:
            for paramName in thisTrial_Loop:
                globals()[paramName] = thisTrial_Loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial_Loop in Trial_Loop:
            Trial_Loop.status = STARTED
            if hasattr(thisTrial_Loop, 'status'):
                thisTrial_Loop.status = STARTED
            currentLoop = Trial_Loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_Loop.rgb)
            if thisTrial_Loop != None:
                for paramName in thisTrial_Loop:
                    globals()[paramName] = thisTrial_Loop[paramName]
            
            # --- Prepare to start Routine "Trial_Session" ---
            # create an object to store info about Routine Trial_Session
            Trial_Session = data.Routine(
                name='Trial_Session',
                components=[RT_EC, RT_EO, RT_Dot_EC, RT_Dot_EO],
            )
            Trial_Session.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            RT_EC.setSound('/Users/tonysaldana/Downloads/Neurotech 25-26/Beginner Project/beep-07a.mp3', secs=1.0, hamming=True)
            RT_EC.setVolume(1.0, log=False)
            RT_EC.seek(0)
            RT_EO.setSound('/Users/tonysaldana/Downloads/Neurotech 25-26/Beginner Project/beep-07a.mp3', secs=1.0, hamming=True)
            RT_EO.setVolume(1.0, log=False)
            RT_EO.seek(0)
            # store start times for Trial_Session
            Trial_Session.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Trial_Session.tStart = globalClock.getTime(format='float')
            Trial_Session.status = STARTED
            thisExp.addData('Trial_Session.started', Trial_Session.tStart)
            Trial_Session.maxDuration = None
            # keep track of which components have finished
            Trial_SessionComponents = Trial_Session.components
            for thisComponent in Trial_Session.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Trial_Session" ---
            Trial_Session.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 34.0:
                # if trial has changed, end Routine now
                if hasattr(thisTrial_Loop, 'status') and thisTrial_Loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *RT_EC* updates
                
                # if RT_EC is starting this frame...
                if RT_EC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    RT_EC.frameNStart = frameN  # exact frame index
                    RT_EC.tStart = t  # local t and not account for scr refresh
                    RT_EC.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('RT_EC.started', tThisFlipGlobal)
                    # update status
                    RT_EC.status = STARTED
                    RT_EC.play(when=win)  # sync with win flip
                
                # if RT_EC is stopping this frame...
                if RT_EC.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > RT_EC.tStartRefresh + 1.0-frameTolerance or RT_EC.isFinished:
                        # keep track of stop time/frame for later
                        RT_EC.tStop = t  # not accounting for scr refresh
                        RT_EC.tStopRefresh = tThisFlipGlobal  # on global time
                        RT_EC.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'RT_EC.stopped')
                        # update status
                        RT_EC.status = FINISHED
                        RT_EC.stop()
                
                # *RT_EO* updates
                
                # if RT_EO is starting this frame...
                if RT_EO.status == NOT_STARTED and tThisFlip >= 17.0-frameTolerance:
                    # keep track of start time/frame for later
                    RT_EO.frameNStart = frameN  # exact frame index
                    RT_EO.tStart = t  # local t and not account for scr refresh
                    RT_EO.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('RT_EO.started', tThisFlipGlobal)
                    # update status
                    RT_EO.status = STARTED
                    RT_EO.play(when=win)  # sync with win flip
                
                # if RT_EO is stopping this frame...
                if RT_EO.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > RT_EO.tStartRefresh + 1.0-frameTolerance or RT_EO.isFinished:
                        # keep track of stop time/frame for later
                        RT_EO.tStop = t  # not accounting for scr refresh
                        RT_EO.tStopRefresh = tThisFlipGlobal  # on global time
                        RT_EO.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'RT_EO.stopped')
                        # update status
                        RT_EO.status = FINISHED
                        RT_EO.stop()
                
                # *RT_Dot_EC* updates
                
                # if RT_Dot_EC is starting this frame...
                if RT_Dot_EC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    RT_Dot_EC.frameNStart = frameN  # exact frame index
                    RT_Dot_EC.tStart = t  # local t and not account for scr refresh
                    RT_Dot_EC.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(RT_Dot_EC, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RT_Dot_EC.started')
                    # update status
                    RT_Dot_EC.status = STARTED
                    RT_Dot_EC.setAutoDraw(True)
                
                # if RT_Dot_EC is active this frame...
                if RT_Dot_EC.status == STARTED:
                    # update params
                    pass
                
                # if RT_Dot_EC is stopping this frame...
                if RT_Dot_EC.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > RT_Dot_EC.tStartRefresh + 17.0-frameTolerance:
                        # keep track of stop time/frame for later
                        RT_Dot_EC.tStop = t  # not accounting for scr refresh
                        RT_Dot_EC.tStopRefresh = tThisFlipGlobal  # on global time
                        RT_Dot_EC.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'RT_Dot_EC.stopped')
                        # update status
                        RT_Dot_EC.status = FINISHED
                        RT_Dot_EC.setAutoDraw(False)
                
                # *RT_Dot_EO* updates
                
                # if RT_Dot_EO is starting this frame...
                if RT_Dot_EO.status == NOT_STARTED and tThisFlip >= 17.0-frameTolerance:
                    # keep track of start time/frame for later
                    RT_Dot_EO.frameNStart = frameN  # exact frame index
                    RT_Dot_EO.tStart = t  # local t and not account for scr refresh
                    RT_Dot_EO.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(RT_Dot_EO, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RT_Dot_EO.started')
                    # update status
                    RT_Dot_EO.status = STARTED
                    RT_Dot_EO.setAutoDraw(True)
                
                # if RT_Dot_EO is active this frame...
                if RT_Dot_EO.status == STARTED:
                    # update params
                    pass
                
                # if RT_Dot_EO is stopping this frame...
                if RT_Dot_EO.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > RT_Dot_EO.tStartRefresh + 17.0-frameTolerance:
                        # keep track of stop time/frame for later
                        RT_Dot_EO.tStop = t  # not accounting for scr refresh
                        RT_Dot_EO.tStopRefresh = tThisFlipGlobal  # on global time
                        RT_Dot_EO.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'RT_Dot_EO.stopped')
                        # update status
                        RT_Dot_EO.status = FINISHED
                        RT_Dot_EO.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Trial_Session,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Trial_Session.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Trial_Session.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Trial_Session" ---
            for thisComponent in Trial_Session.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Trial_Session
            Trial_Session.tStop = globalClock.getTime(format='float')
            Trial_Session.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Trial_Session.stopped', Trial_Session.tStop)
            RT_EC.pause()  # ensure sound has stopped at end of Routine
            RT_EO.pause()  # ensure sound has stopped at end of Routine
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if Trial_Session.maxDurationReached:
                routineTimer.addTime(-Trial_Session.maxDuration)
            elif Trial_Session.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-34.000000)
            # mark thisTrial_Loop as finished
            if hasattr(thisTrial_Loop, 'status'):
                thisTrial_Loop.status = FINISHED
            # if awaiting a pause, pause now
            if Trial_Loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                Trial_Loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'Trial_Loop'
        Trial_Loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if Trial_Loop.trialList in ([], [None], None):
            params = []
        else:
            params = Trial_Loop.trialList[0].keys()
        # save data for this loop
        Trial_Loop.saveAsText(filename + '_Trial_Loop.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "rest" ---
        # create an object to store info about Routine rest
        rest = data.Routine(
            name='rest',
            components=[Rest_text, Rest_Key_Resp],
        )
        rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for Rest_Key_Resp
        Rest_Key_Resp.keys = []
        Rest_Key_Resp.rt = []
        _Rest_Key_Resp_allKeys = []
        # store start times for rest
        rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest.tStart = globalClock.getTime(format='float')
        rest.status = STARTED
        thisExp.addData('rest.started', rest.tStart)
        rest.maxDuration = None
        # keep track of which components have finished
        restComponents = rest.components
        for thisComponent in rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rest" ---
        rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_Loop, 'status') and thisBlock_Loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Rest_text* updates
            
            # if Rest_text is starting this frame...
            if Rest_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Rest_text.frameNStart = frameN  # exact frame index
                Rest_text.tStart = t  # local t and not account for scr refresh
                Rest_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Rest_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rest_text.started')
                # update status
                Rest_text.status = STARTED
                Rest_text.setAutoDraw(True)
            
            # if Rest_text is active this frame...
            if Rest_text.status == STARTED:
                # update params
                pass
            
            # *Rest_Key_Resp* updates
            waitOnFlip = False
            
            # if Rest_Key_Resp is starting this frame...
            if Rest_Key_Resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Rest_Key_Resp.frameNStart = frameN  # exact frame index
                Rest_Key_Resp.tStart = t  # local t and not account for scr refresh
                Rest_Key_Resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Rest_Key_Resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rest_Key_Resp.started')
                # update status
                Rest_Key_Resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(Rest_Key_Resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(Rest_Key_Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if Rest_Key_Resp.status == STARTED and not waitOnFlip:
                theseKeys = Rest_Key_Resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _Rest_Key_Resp_allKeys.extend(theseKeys)
                if len(_Rest_Key_Resp_allKeys):
                    Rest_Key_Resp.keys = _Rest_Key_Resp_allKeys[-1].name  # just the last key pressed
                    Rest_Key_Resp.rt = _Rest_Key_Resp_allKeys[-1].rt
                    Rest_Key_Resp.duration = _Rest_Key_Resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=rest,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest
        rest.tStop = globalClock.getTime(format='float')
        rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest.stopped', rest.tStop)
        # check responses
        if Rest_Key_Resp.keys in ['', [], None]:  # No response was made
            Rest_Key_Resp.keys = None
        Block_Loop.addData('Rest_Key_Resp.keys',Rest_Key_Resp.keys)
        if Rest_Key_Resp.keys != None:  # we had a response
            Block_Loop.addData('Rest_Key_Resp.rt', Rest_Key_Resp.rt)
            Block_Loop.addData('Rest_Key_Resp.duration', Rest_Key_Resp.duration)
        # the Routine "rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisBlock_Loop as finished
        if hasattr(thisBlock_Loop, 'status'):
            thisBlock_Loop.status = FINISHED
        # if awaiting a pause, pause now
        if Block_Loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            Block_Loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'Block_Loop'
    Block_Loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if Block_Loop.trialList in ([], [None], None):
        params = []
    else:
        params = Block_Loop.trialList[0].keys()
    # save data for this loop
    Block_Loop.saveAsText(filename + '_Block_Loop.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
